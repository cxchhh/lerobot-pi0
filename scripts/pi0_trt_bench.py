"""Numerical alignment bench for PI0 TensorRT integration.

Two check modes:

    --check wrappers     Compare native `sample_actions` (base branch) against
                         PrefillWrapper + DiffusionStepWrapper composed by
                         hand. Pure torch, no TRT. Regression gate.

    --check trt          (After Phase A engines are built) compare torch
                         vs TRT-backed sample_actions on real inputs.
                         Prints max_abs / max_rel per module.

Same CLI as ``server.py`` (uses ``TrainPipelineConfig`` + ``parser.wrap``).
Example:

    python scripts/pi0_trt_bench.py --check wrappers \
        --policy.path=outputs/train/.../pretrained_model \
        --dataset.repo_id=lerobot_data/bfm-v1.9 \
        --dataset.root=./lerobot_data/bfm-v1.9
"""

import logging
import os
import sys
from dataclasses import asdict
from pprint import pformat

import torch
from termcolor import colored

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pi0.modeling_pi0 import PI0FlowMatching, PI0Policy
from lerobot.common.policies.pi0.trt_wrappers import (
    DiffusionStepWrapper,
    PrefillWrapper,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


def _fake_batch(policy: PI0Policy, device: str, seed: int = 0) -> dict:
    """Deterministic random batch matching the policy's input spec."""
    torch.manual_seed(seed)
    batch = {}
    for key, feat in policy.config.image_features.items():
        c, h, w = feat.shape
        batch[key] = torch.rand(1, c, h, w, dtype=torch.float32, device=device)
    state_dim = policy.config.robot_state_feature.shape[0]
    batch["observation.state"] = torch.randn(1, state_dim, device=device)
    batch["task"] = ["walk forward"]
    batch["task_index"] = torch.tensor([0], device=device)
    return batch


def _prep_inputs(policy: PI0Policy, batch: dict):
    """Run the pre-`sample_actions` pipeline of PI0Policy.get_action_chunk."""
    batch = policy.normalize_inputs(batch)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    return images, img_masks, state, lang_tokens, lang_masks


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    diff = (a32 - b32).abs()
    denom = a32.abs().clamp_min(1e-6)
    rel = (diff / denom).max().item()
    print(
        f"  {name:32s}  max_abs={diff.max().item():.3e}  "
        f"mean_abs={diff.mean().item():.3e}  max_rel={rel:.3e}  shape={tuple(a.shape)}"
    )


def _native_prefill(flow: PI0FlowMatching, images, img_masks, lang_tokens, lang_masks):
    """Run just the native prefill and return (past_kv_dict, prefix_pad_masks)."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    from lerobot.common.policies.pi0.modeling_pi0 import make_att_2d_masks
    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_kv = flow.paligemma_with_expert.forward(
        attention_mask=att_2d,
        position_ids=pos_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=flow.config.use_cache,
        fill_kv_cache=True,
    )
    return past_kv, prefix_pad_masks


def _stack_native_kv(past_kv_dict) -> torch.Tensor:
    """Convert native past_key_values dict (layer_idx -> {k, v}) to stacked
    tensor of shape (L, 2, B, T, KV, D)."""
    layers = sorted(past_kv_dict.keys())
    ks = torch.stack([past_kv_dict[i]["key_states"] for i in layers], dim=0)
    vs = torch.stack([past_kv_dict[i]["value_states"] for i in layers], dim=0)
    return torch.stack([ks, vs], dim=1)


@torch.no_grad()
def check_wrappers(policy: PI0Policy, device: str) -> None:
    flow: PI0FlowMatching = policy.model
    prefill = PrefillWrapper(flow).eval()
    step = DiffusionStepWrapper(flow).eval()

    batch = _fake_batch(policy, device)
    images, img_masks, state, lang_tokens, lang_masks = _prep_inputs(policy, batch)

    img_keys = [k for k in policy.config.image_features if k in batch]
    print(f"[wrappers] cameras used (in order): {img_keys}")
    if len(images) < 2:
        raise RuntimeError(f"Wrapper assumes 2 cameras, got {len(images)}.")
    img_head, img_right = images[0], images[1]
    mask_head, mask_right = img_masks[0], img_masks[1]

    # === Level 1: prefill ===
    native_pkv_dict, native_prefix_pad = _native_prefill(flow, images, img_masks, lang_tokens, lang_masks)
    native_pkv_stack = _stack_native_kv(native_pkv_dict)
    wrapper_pkv, wrapper_prefix_pad = prefill(
        img_head, img_right, mask_head, mask_right, lang_tokens, lang_masks
    )
    print(colored("\n=== Level 1: prefill past_kv (wrapper vs native) ===", "yellow", attrs=["bold"]))
    _report("past_kv", wrapper_pkv, native_pkv_stack)
    _report("prefix_pad_masks_i32", wrapper_prefix_pad.to(torch.int32), native_prefix_pad.to(torch.int32))

    # === Level 2: single denoise step, using SAME past_kv ===
    torch.manual_seed(42)
    bsize = state.shape[0]
    n_act = flow.config.n_action_steps
    noise = flow.sample_noise((bsize, n_act, flow.config.max_action_dim), device)
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    expanded_time = time.expand(bsize, n_act)

    # Per-layer hooks on the expert to capture native + wrapper hidden state at every layer.
    ge_layers = flow.paligemma_with_expert.gemma_expert.model.layers
    native_hidden_out = [None] * len(ge_layers)
    # Layer-0 sub-step diff points (input_layernorm out, q_proj out, o_proj out, mlp out)
    native_l0 = {}
    wrapper_l0 = {}

    def _make_capture(idx, sink):
        def hk(mod, inp, out):
            sink[idx] = out.detach().clone()
        return hk

    def _make_l0_capture(name, sink):
        def hk(mod, inp, out):
            sink[name] = out.detach().clone()
        return hk

    hs = [ge_layers[i].mlp.register_forward_hook(_make_capture(i, native_hidden_out)) for i in range(len(ge_layers))]
    l0 = ge_layers[0]
    hs.append(l0.input_layernorm.register_forward_hook(_make_l0_capture("in_ln", native_l0)))
    hs.append(l0.self_attn.q_proj.register_forward_hook(_make_l0_capture("q_proj", native_l0)))
    hs.append(l0.self_attn.o_proj.register_forward_hook(_make_l0_capture("o_proj", native_l0)))
    hs.append(l0.post_attention_layernorm.register_forward_hook(_make_l0_capture("post_ln", native_l0)))
    hs.append(l0.mlp.register_forward_hook(_make_l0_capture("mlp", native_l0)))
    try:
        native_v = flow._denoise_step_base(state, native_prefix_pad, native_pkv_dict, x_t, expanded_time)
    finally:
        for h in hs:
            h.remove()

    wrapper_hidden_out = [None] * len(ge_layers)
    hs = [ge_layers[i].mlp.register_forward_hook(_make_capture(i, wrapper_hidden_out)) for i in range(len(ge_layers))]
    hs.append(l0.input_layernorm.register_forward_hook(_make_l0_capture("in_ln", wrapper_l0)))
    hs.append(l0.self_attn.q_proj.register_forward_hook(_make_l0_capture("q_proj", wrapper_l0)))
    hs.append(l0.self_attn.o_proj.register_forward_hook(_make_l0_capture("o_proj", wrapper_l0)))
    hs.append(l0.post_attention_layernorm.register_forward_hook(_make_l0_capture("post_ln", wrapper_l0)))
    hs.append(l0.mlp.register_forward_hook(_make_l0_capture("mlp", wrapper_l0)))
    try:
        wrapper_v = step(state, x_t, expanded_time, native_pkv_stack, native_prefix_pad)
    finally:
        for h in hs:
            h.remove()

    print(colored("\n=== Level 2: single denoise v_t (SAME past_kv) ===", "yellow", attrs=["bold"]))
    _report("v_t_step0", wrapper_v, native_v)
    print(colored("--- layer 0 sub-step diff ---", "yellow"))
    for k in ("in_ln", "q_proj", "o_proj", "post_ln", "mlp"):
        _report(f"l0.{k}", wrapper_l0[k], native_l0[k])
    print(colored("--- per-layer expert output diff (post-mlp) ---", "yellow"))
    for i in range(len(ge_layers)):
        # gemma layer outputs may be tuples; unwrap
        n_out = native_hidden_out[i][0] if isinstance(native_hidden_out[i], tuple) else native_hidden_out[i]
        w_out = wrapper_hidden_out[i][0] if isinstance(wrapper_hidden_out[i], tuple) else wrapper_hidden_out[i]
        _report(f"layer{i:02d}", w_out, n_out)

    # === Level 3: full 10-step loop, both paths from scratch ===
    native_actions = flow.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
    )
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    dt = -1.0 / flow.config.num_steps
    while time >= -dt / 2:
        expanded_time = time.expand(bsize, n_act)
        v_t = step(state, x_t, expanded_time, wrapper_pkv, wrapper_prefix_pad)
        x_t = x_t + dt * v_t
        time = time + dt
    print(colored("\n=== Level 3: full 10-step chunk (wrapper vs native) ===", "yellow", attrs=["bold"]))
    _report("final_actions", x_t, native_actions)


def check_trt(policy: PI0Policy, device: str) -> None:
    """Build (or load cached) TRT engines from the policy's ckpt dir, then
    diff torch-native vs TRT-backed sample_actions on a fixed batch.
    Same CLI as check_wrappers — reads ckpt path from parser.wrap cfg."""
    try:
        import tensorrt  # noqa: F401
    except ModuleNotFoundError:
        print(colored("tensorrt not installed", "red"))
        return

    from pathlib import Path

    from lerobot.common.policies.pi0.trt_infer import PI0TRTBackend

    ckpt_dir = Path(os.environ.get("PI0_TRT_CKPT_DIR", ""))
    if not ckpt_dir.exists():
        raise RuntimeError(
            "Set PI0_TRT_CKPT_DIR=<policy ckpt dir> (the same path passed as --policy.path)"
        )

    precision = os.environ.get("PI0_TRT_PRECISION", "bf16")
    rtc_mode = os.environ.get("PI0_TRT_RTC_MODE", "torch")
    print(colored(f"[trt] ckpt_dir={ckpt_dir} precision={precision} rtc_mode={rtc_mode}", "cyan"))

    backend = PI0TRTBackend(
        torch_policy=policy,
        ckpt_dir=ckpt_dir,
        precision=precision,
        rtc_mode=rtc_mode,
        debug=True,
    )

    # End-to-end diff: torch vs TRT sample_actions on the same noise/batch.
    flow = policy.model
    batch = _fake_batch(policy, device)
    images, img_masks, state, lang_tokens, lang_masks = _prep_inputs(policy, batch)
    torch.manual_seed(42)
    bsize = state.shape[0]
    n_act = flow.config.n_action_steps
    noise = flow.sample_noise((bsize, n_act, flow.config.max_action_dim), device)

    # Torch baseline (no backend attached to shadow policy — but same weights)
    with torch.no_grad():
        native = flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())

    # Attach backend, run TRT path
    flow.trt_backend = backend
    flow.trt_rtc_mode = rtc_mode
    with torch.no_grad():
        trt_out = flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())

    print(colored("\n=== TRT vs Native (end-to-end sample_actions) ===", "yellow", attrs=["bold"]))
    _report("final_actions", trt_out, native)

    # --- Perf benchmark: base (no action_prefix) vs RTC (with action_prefix) ---
    import time
    warmup, iters = 10, 50
    print(colored("\n=== perf: sample_actions (warm-up 10, timed 50) ===", "yellow", attrs=["bold"]))

    # RTC needs a plausible action_prefix. Use a plain slice of the ideal action
    # from the torch native rollout — matches how the client would feed it.
    prefix_len = 10  # typical --rtc-prefix-len
    action_prefix = native[:, :prefix_len].detach().clone()

    def _time_path(label, run_fn):
        # RTC uses autograd internally — cannot use torch.no_grad. Just time.
        for _ in range(warmup):
            _ = run_fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = run_fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1000
        print(f"  {label:<28s} {dt:.2f} ms / call")
        return dt

    # ---- base path (no action_prefix) ----
    flow.trt_backend = None
    with torch.no_grad():
        ms_torch_base = _time_path("torch native | base",
            lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise))
    flow.trt_backend = backend
    with torch.no_grad():
        ms_trt_base = _time_path("TRT backend  | base",
            lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise))

    # ---- RTC path (with action_prefix) ----
    flow.trt_backend = None
    ms_torch_rtc = _time_path("torch native | RTC",
        lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise,
                                    action_prefix=action_prefix, rtc_inference_delay=1,
                                    rtc_prefix_attention_horizon=prefix_len,
                                    rtc_max_guidance_weight=1.0))
    flow.trt_backend = backend
    flow.trt_rtc_mode = rtc_mode
    ms_trt_rtc = _time_path("TRT backend  | RTC",
        lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise,
                                    action_prefix=action_prefix, rtc_inference_delay=1,
                                    rtc_prefix_attention_horizon=prefix_len,
                                    rtc_max_guidance_weight=1.0))

    # ---- VLM-only TRT (diffusion stays in torch) ----
    flow.trt_backend = backend
    flow.trt_vlm_only = True
    with torch.no_grad():
        ms_vlm_base = _time_path("TRT vlm-only | base",
            lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise))
    ms_vlm_rtc = _time_path("TRT vlm-only | RTC",
        lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise,
                                    action_prefix=action_prefix, rtc_inference_delay=1,
                                    rtc_prefix_attention_horizon=prefix_len,
                                    rtc_max_guidance_weight=1.0))
    flow.trt_vlm_only = False

    # ---- diffusion-only TRT (vlm stays in torch — highest precision) ----
    flow.trt_diffusion_only = True
    with torch.no_grad():
        ms_diff_base = _time_path("TRT diff-only| base",
            lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise))
    ms_diff_rtc = _time_path("TRT diff-only| RTC",
        lambda: flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise,
                                    action_prefix=action_prefix, rtc_inference_delay=1,
                                    rtc_prefix_attention_horizon=prefix_len,
                                    rtc_max_guidance_weight=1.0))
    # Correctness diff for diffusion-only mode
    with torch.no_grad():
        diff_only_out = flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
    print(colored("--- accuracy (diff-only vs native torch) ---", "yellow"))
    _report("final_actions[diff-only]", diff_only_out, native)
    flow.trt_diffusion_only = False

    print(colored("\n=== summary ===", "green", attrs=["bold"]))
    print(f"  base:  torch {ms_torch_base:.2f}  |  full-TRT {ms_trt_base:.2f} ({ms_torch_base/ms_trt_base:.2f}x)")
    print(f"                                    |  vlm-only {ms_vlm_base:.2f} ({ms_torch_base/ms_vlm_base:.2f}x)")
    print(f"                                    |  diff-only {ms_diff_base:.2f} ({ms_torch_base/ms_diff_base:.2f}x)")
    print(f"  RTC :  torch {ms_torch_rtc:.2f}  |  full-TRT {ms_trt_rtc:.2f} ({ms_torch_rtc/ms_trt_rtc:.2f}x)")
    print(f"                                    |  vlm-only {ms_vlm_rtc:.2f} ({ms_torch_rtc/ms_vlm_rtc:.2f}x)")
    print(f"                                    |  diff-only {ms_diff_rtc:.2f} ({ms_torch_rtc/ms_diff_rtc:.2f}x)")


@parser.wrap()
def _main(cfg: TrainPipelineConfig):
    cfg.validate()
    cfg.policy.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg.policy.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(pformat(asdict(cfg)))
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info("load dataset metainfo")
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    logging.info("Making policy.")
    if cfg.policy.type == "hvla":
        cfg.policy.eval = True
        cfg.policy.load_path = cfg.policy.pretrained_path

    network: PreTrainedPolicy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    network.eval()

    check = os.environ.get("PI0_TRT_CHECK", "wrappers")
    print(colored(f"\n>>> running check = {check!r} <<<\n", "cyan", attrs=["bold"]))
    if check == "wrappers":
        check_wrappers(network, device)
    elif check == "trt":
        check_trt(network, device)
    else:
        raise ValueError(f"Unknown check: {check}. Use wrappers or trt.")


if __name__ == "__main__":
    init_logging()
    _main()
