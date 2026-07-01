"""TensorRT runtime for PI0 inference.

Splits the policy into two engines (VLM prefill + one diffusion step) and
runs the flow-matching outer loop in Python. On first launch it exports the
two ONNX modules from :mod:`trt_wrappers`, builds bf16 engines, caches the
``.plan`` files, and records ``build_info.json`` for invalidation.

Public API (used by ``modeling_pi0.py`` when a backend is attached):

    backend.vlm_prefill(images, img_masks, lang_tokens, lang_masks)
        -> (past_kv_stack, prefix_pad_masks)
    backend.diffusion_step(state, prefix_pad_masks, past_kv_stack, x_t, timestep)
        -> v_t

RTC (F3, still experimental) exposes:

    backend.diffusion_step_and_vjp(state, prefix_pad, past_kv, x_t, t, error)
        -> (v_t, grad_x_error)

If the F3 backward engine cannot be built, this raises ``NotSupportedError``
and the caller must fall back to PyTorch autograd.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from lerobot.common.policies.pi0.trt_wrappers import (
    DiffusionStepBwdWrapper,
    DiffusionStepWrapper,
    PrefillWrapper,
)

log = logging.getLogger(__name__)


class NotSupportedError(RuntimeError):
    """RTC bwd engine unavailable (export/build failed)."""


# TensorRT is imported lazily so `import lerobot.common.policies.pi0.trt_infer`
# doesn't require TRT to be installed at module-load time. TRT 10 accepts a
# raw CUDA stream handle for execute_async_v3, so we don't need pycuda — we
# feed torch.cuda.current_stream().cuda_stream directly.
_trt = None


def _lazy_trt():
    global _trt
    if _trt is not None:
        return _trt
    import tensorrt as trt  # noqa: PLC0415
    _trt = trt
    return _trt


# -----------------------------------------------------------------------------
# Precision / dtype mapping
# -----------------------------------------------------------------------------

_TORCH_TO_TRT_DTYPE = None
_TORCH_TO_NP_DTYPE = {
    torch.float32: np.float32,
    torch.bfloat16: None,  # numpy has no bf16; view as uint16 for host-side buffering
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}


def _torch_to_trt_dtype(dtype: torch.dtype):
    global _TORCH_TO_TRT_DTYPE
    if _TORCH_TO_TRT_DTYPE is None:
        trt = _lazy_trt()
        _TORCH_TO_TRT_DTYPE = {
            torch.float32: trt.float32,
            torch.bfloat16: trt.bfloat16,
            torch.float16: trt.float16,
            torch.int64: trt.int64,
            torch.int32: trt.int32,
            torch.bool: trt.bool,
        }
    return _TORCH_TO_TRT_DTYPE[dtype]


# -----------------------------------------------------------------------------
# Build info (cache invalidation)
# -----------------------------------------------------------------------------

@dataclass
class BuildInfo:
    torch_version: str
    trt_version: str
    gpu_name: str
    precision: str
    prefill_input_shapes: dict = field(default_factory=dict)
    diffusion_input_shapes: dict = field(default_factory=dict)
    diffusion_bwd_input_shapes: dict = field(default_factory=dict)
    has_diffusion_bwd: bool = False

    @classmethod
    def from_env(cls, precision: str) -> "BuildInfo":
        trt = _lazy_trt()
        return cls(
            torch_version=torch.__version__,
            trt_version=trt.__version__,
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            precision=precision,
        )

    def matches(self, other: "BuildInfo") -> bool:
        return (
            self.torch_version == other.torch_version
            and self.trt_version == other.trt_version
            and self.gpu_name == other.gpu_name
            and self.precision == other.precision
            and self.prefill_input_shapes == other.prefill_input_shapes
            and self.diffusion_input_shapes == other.diffusion_input_shapes
        )


# -----------------------------------------------------------------------------
# ONNX export
# -----------------------------------------------------------------------------

def _dummy_prefill_inputs(flow, device: str) -> tuple:
    """Zeros in the shapes/dtypes the wrapper expects."""
    C, H, W = 3, 224, 224
    img_head = torch.zeros(1, C, H, W, dtype=torch.bfloat16, device=device)
    img_right = torch.zeros(1, C, H, W, dtype=torch.bfloat16, device=device)
    mask_head = torch.ones(1, dtype=torch.bool, device=device)
    mask_right = torch.ones(1, dtype=torch.bool, device=device)
    L = flow.config.tokenizer_max_length
    lang_tokens = torch.zeros(1, L, dtype=torch.int64, device=device)
    lang_masks = torch.ones(1, L, dtype=torch.bool, device=device)
    return img_head, img_right, mask_head, mask_right, lang_tokens, lang_masks


def _dummy_diffusion_inputs(flow, prefix_len: int, device: str) -> tuple:
    n_layers = flow.config.use_cache and flow.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers or 18
    kv_heads = flow.paligemma_with_expert.paligemma.config.text_config.num_key_value_heads
    head_dim = flow.paligemma_with_expert.paligemma.config.text_config.head_dim
    n_act = flow.config.n_action_steps
    state = torch.zeros(1, flow.config.max_state_dim, dtype=torch.float32, device=device)
    x_t = torch.zeros(1, n_act, flow.config.max_action_dim, dtype=torch.float32, device=device)
    timestep = torch.ones(1, n_act, dtype=torch.float32, device=device)
    past_kv = torch.zeros(n_layers, 2, 1, prefix_len, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    prefix_pad = torch.ones(1, prefix_len, dtype=torch.bool, device=device)
    return state, x_t, timestep, past_kv, prefix_pad


def _export_onnx(module: torch.nn.Module, dummy: tuple, path: Path, input_names: list[str], output_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        # torch 2.7 dynamo exporter — dynamic_shapes left None (all shapes static).
        torch.onnx.export(
            module,
            dummy,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            opset_version=17,
            verbose=False,
        )
    log.info("[trt] exported %s (%.1f MB)", path.name, path.stat().st_size / 2**20)


# -----------------------------------------------------------------------------
# Engine build + load
# -----------------------------------------------------------------------------

def _build_engine(onnx_path: Path, plan_path: Path, precision: str) -> None:
    trt = _lazy_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 11 removed legacy precision flags (BF16 / FP16 / PREFER_PRECISION_CONSTRAINTS).
    # Precision is now derived from the ONNX graph itself via STRONGLY_TYPED network.
    # Our exported ONNX already has bf16 tensors on the transformer layers and fp32
    # around softmax / final projections, so this preserves the intended mixed layout.
    # The `precision` argument stays only for cache-invalidation bookkeeping.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(str(onnx_path))
    if not ok:
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    # 1 GB workspace — enough for our layer sizes; leaves the rest of VRAM for
    # weights (up to ~6 GB per engine) and TRT tactic scratch. Larger values
    # trigger CUDA OOM when the diffusion engine follows the prefill build in
    # the same process on <= 16 GB free.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    log.info("[trt] building %s (precision=%s from ONNX, %d layers) — may take minutes...",
             plan_path.name, precision, network.num_layers)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TRT build failed for {onnx_path}")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(serialized)
    log.info("[trt] saved %s (%.1f MB)", plan_path.name, plan_path.stat().st_size / 2**20)


def _load_engine(plan_path: Path):
    trt = _lazy_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {plan_path}")
    return engine


# -----------------------------------------------------------------------------
# Engine execution
# -----------------------------------------------------------------------------

class _EngineRunner:
    """Binds torch tensors to a TRT execution context and runs it."""

    def __init__(self, engine, input_names: list[str], output_dtypes: dict[str, torch.dtype], output_shapes: dict[str, tuple]):
        trt = _lazy_trt()
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_names = input_names
        # Pre-allocate outputs
        self.outputs: dict[str, torch.Tensor] = {}
        for name, dtype in output_dtypes.items():
            shape = output_shapes[name]
            t = torch.empty(shape, dtype=dtype, device="cuda")
            self.outputs[name] = t
        self.stream = torch.cuda.Stream()

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Set input tensors + addresses
        for name in self.input_names:
            t = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, int(t.data_ptr()))
        for name, out in self.outputs.items():
            self.context.set_tensor_address(name, int(out.data_ptr()))
        # Execute on our stream, then sync
        stream_handle = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")
        torch.cuda.current_stream().synchronize()
        # Return clones so the caller can safely reuse the output buffers across calls
        return {k: v.clone() for k, v in self.outputs.items()}


# -----------------------------------------------------------------------------
# PI0TRTBackend
# -----------------------------------------------------------------------------

_PREFILL_IN = ["img_head", "img_right", "img_head_mask", "img_right_mask", "lang_tokens", "lang_masks"]
_PREFILL_OUT = ["past_kv", "prefix_pad"]

_DIFFUSION_IN = ["state", "x_t", "timestep", "past_kv", "prefix_pad"]
_DIFFUSION_OUT = ["v_t"]

_DIFFUSION_BWD_IN = _DIFFUSION_IN + ["error"]
_DIFFUSION_BWD_OUT = ["grad_x"]


class PI0TRTBackend:
    """Encapsulates TRT engines for PI0's VLM prefill and diffusion single step.

    Instantiated once by ``server.py`` when ``--trt`` is passed, attached to
    ``PI0FlowMatching.trt_backend``. ``sample_actions`` / ``_denoise_step_base``
    /``_denoise_step_rtc`` in ``modeling_pi0.py`` dispatch here.
    """

    def __init__(
        self,
        torch_policy,
        ckpt_dir: Path,
        precision: str = "bf16",
        rtc_mode: str = "torch",
        debug: bool = False,
    ):
        self.torch_policy = torch_policy
        self.flow = torch_policy.model
        self.ckpt_dir = Path(ckpt_dir)
        self.cache_dir = self.ckpt_dir / "trt_cache"
        self.precision = precision
        self.rtc_mode = rtc_mode
        self.debug = debug
        self.device = "cuda"

        # Shapes we're going to bake in — batch=1, 2 cameras, 48 tokens, 50 steps
        cfg = self.flow.config
        text_cfg = self.flow.paligemma_with_expert.paligemma.config.text_config
        self.n_layers = text_cfg.num_hidden_layers
        self.kv_heads = text_cfg.num_key_value_heads
        self.head_dim = text_cfg.head_dim
        self.n_act = cfg.n_action_steps
        self.max_state_dim = cfg.max_state_dim
        self.max_action_dim = cfg.max_action_dim
        # prefix_len is 2*num_image_tokens + tokenizer_max_length (batch=1, 2 cams)
        self.prefix_len = 2 * text_cfg.num_image_tokens + cfg.tokenizer_max_length

        self._prepare_engines()

        if debug:
            self._sanity_check()

    # ----- Cache prep -----

    def _prepare_engines(self):
        _lazy_trt()  # trigger import + fail early if not installed
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        info = BuildInfo.from_env(self.precision)
        info.prefill_input_shapes = {
            "img_head": [1, 3, 224, 224], "img_right": [1, 3, 224, 224],
            "img_head_mask": [1], "img_right_mask": [1],
            "lang_tokens": [1, self.flow.config.tokenizer_max_length],
            "lang_masks": [1, self.flow.config.tokenizer_max_length],
        }
        info.diffusion_input_shapes = {
            "state": [1, self.max_state_dim],
            "x_t": [1, self.n_act, self.max_action_dim],
            "timestep": [1, self.n_act],
            "past_kv": [self.n_layers, 2, 1, self.prefix_len, self.kv_heads, self.head_dim],
            "prefix_pad": [1, self.prefix_len],
        }
        rebuild = self._check_and_rebuild(info)

        onnx_p = self.cache_dir / "vlm_prefix.onnx"
        plan_p = self.cache_dir / f"vlm_prefix_{self.precision}.plan"
        onnx_d = self.cache_dir / "diffusion_step.onnx"
        plan_d = self.cache_dir / f"diffusion_step_{self.precision}.plan"
        onnx_b = self.cache_dir / "diffusion_step_bwd.onnx"
        plan_b = self.cache_dir / f"diffusion_step_bwd_{self.precision}.plan"

        # --- Phase 1: build all missing engines ---
        # TRT autotuner needs several GB of GPU workspace, more than we have
        # once the torch policy weights (~4 GB paligemma+expert bf16) are also
        # resident. Strategy per engine:
        #   1. Ensure torch policy on GPU (needed for dynamo export tracing)
        #   2. Export ONNX if missing
        #   3. Move policy to CPU so TRT has full VRAM to search tactics
        #   4. Build engine
        #   5. Sweep and move policy back (in case another engine needs export)
        import gc

        def _sweep():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _to_cpu():
            self.flow.to("cpu")
            _sweep()

        def _to_gpu():
            self.flow.to(self.device)

        def _build_one(need_export: bool, exporter, onnx_path, plan_path):
            if need_export:
                _to_gpu()
                exporter(onnx_path)
            _to_cpu()
            _build_engine(onnx_path, plan_path, self.precision)
            _sweep()

        if rebuild or not plan_p.exists():
            _build_one(
                need_export=(rebuild or not plan_p.exists() and not onnx_p.exists()) or rebuild,
                exporter=self._export_prefill,
                onnx_path=onnx_p,
                plan_path=plan_p,
            )
            # Persist immediately so a later failure doesn't invalidate this plan.
            self._save_build_info(info)
        if rebuild or not plan_d.exists():
            _build_one(
                need_export=(rebuild or not onnx_d.exists()),
                exporter=self._export_diffusion,
                onnx_path=onnx_d,
                plan_path=plan_d,
            )
            self._save_build_info(info)

        # Optional RTC bwd (F3)
        info.has_diffusion_bwd = False
        bwd_ok = False
        if self.rtc_mode in ("split", "tensorrt-only"):
            try:
                if rebuild or not plan_b.exists():
                    _build_one(
                        need_export=(rebuild or not onnx_b.exists()),
                        exporter=self._export_diffusion_bwd,
                        onnx_path=onnx_b,
                        plan_path=plan_b,
                    )
                bwd_ok = True
            except Exception as e:
                if self.rtc_mode == "tensorrt-only":
                    raise
                log.warning(
                    "[trt] RTC bwd engine unavailable (%s); RTC steps will fall back to PyTorch autograd.",
                    e.__class__.__name__ + ": " + str(e)[:200],
                )

        # --- Phase 2: load engines + create runners ---
        # Deserializing a 5+GB plan needs contiguous VRAM — if the torch
        # policy weights (4 GB) are still resident, the allocator fragments
        # and fails. Move policy to CPU for load, restore afterwards.
        log.info("[trt] loading engines (policy → CPU for contiguous alloc)")
        _to_cpu()
        self._prefill_engine = _load_engine(plan_p)
        self._prefill_runner = _EngineRunner(
            self._prefill_engine,
            _PREFILL_IN,
            output_dtypes={"past_kv": torch.bfloat16, "prefix_pad": torch.bool},
            output_shapes={
                "past_kv": (self.n_layers, 2, 1, self.prefix_len, self.kv_heads, self.head_dim),
                "prefix_pad": (1, self.prefix_len),
            },
        )
        self._diffusion_engine = _load_engine(plan_d)
        self._diffusion_runner = _EngineRunner(
            self._diffusion_engine,
            _DIFFUSION_IN,
            output_dtypes={"v_t": torch.float32},
            output_shapes={"v_t": (1, self.n_act, self.max_action_dim)},
        )
        self._diffusion_bwd_runner = None
        if bwd_ok:
            self._diffusion_bwd_engine = _load_engine(plan_b)
            self._diffusion_bwd_runner = _EngineRunner(
                self._diffusion_bwd_engine,
                _DIFFUSION_BWD_IN,
                output_dtypes={"grad_x": torch.float32},
                output_shapes={"grad_x": (1, self.n_act, self.max_action_dim)},
            )
            info.has_diffusion_bwd = True
            log.info("[trt] RTC bwd engine ready (mode=%s)", self.rtc_mode)

        # Restore torch policy to GPU — needed for RTC fallback and sanity check.
        log.info("[trt] restoring torch policy to %s", self.device)
        _to_gpu()

        # Persist build info
        self._save_build_info(info)

    def _check_and_rebuild(self, info: BuildInfo) -> bool:
        info_path = self.cache_dir / "build_info.json"
        if not info_path.exists():
            # Missing build_info can also mean a prior build got killed after
            # some plans were written. Treat cache as valid if no plans exist
            # at all (fresh install) or if we can't verify — only rebuild when
            # the file is present AND signature mismatches.
            return False
        try:
            existing = BuildInfo(**json.loads(info_path.read_text()))
        except Exception:
            return True
        matches = info.matches(existing)
        if not matches:
            log.warning("[trt] build_info mismatch — will rebuild engines.")
        return not matches

    def _save_build_info(self, info: BuildInfo):
        (self.cache_dir / "build_info.json").write_text(json.dumps(asdict(info), indent=2))

    # ----- Export -----

    def _export_prefill(self, path: Path):
        wrapper = PrefillWrapper(self.flow).eval()
        dummy = _dummy_prefill_inputs(self.flow, self.device)
        _export_onnx(wrapper, dummy, path, _PREFILL_IN, _PREFILL_OUT)

    def _export_diffusion(self, path: Path):
        wrapper = DiffusionStepWrapper(self.flow).eval()
        dummy = _dummy_diffusion_inputs(self.flow, self.prefix_len, self.device)
        _export_onnx(wrapper, dummy, path, _DIFFUSION_IN, _DIFFUSION_OUT)

    def _export_diffusion_bwd(self, path: Path):
        fwd = DiffusionStepWrapper(self.flow).eval()
        wrapper = DiffusionStepBwdWrapper(fwd)
        dummy = _dummy_diffusion_inputs(self.flow, self.prefix_len, self.device)
        error = torch.zeros(1, self.n_act, self.max_action_dim, dtype=torch.float32, device=self.device)
        # torch.autograd.grad inside forward needs grad-enabled trace.
        # Try dynamo export first — it may reject the autograd op.
        with torch.enable_grad():
            torch.onnx.export(
                wrapper,
                (*dummy, error),
                str(path),
                input_names=_DIFFUSION_BWD_IN,
                output_names=_DIFFUSION_BWD_OUT,
                dynamo=True,
                opset_version=17,
                verbose=False,
            )

    # ----- Public runtime API -----

    def vlm_prefill(self, images, img_masks, lang_tokens, lang_masks):
        """Run prefix engine. `images`/`img_masks` are the lists produced by
        ``PI0Policy.prepare_images`` (in policy.config.image_features order)."""
        if len(images) < 2:
            raise RuntimeError(f"TRT backend expects 2 cameras, got {len(images)}")
        inputs = {
            "img_head": images[0].to(torch.bfloat16),
            "img_right": images[1].to(torch.bfloat16),
            "img_head_mask": img_masks[0],
            "img_right_mask": img_masks[1],
            "lang_tokens": lang_tokens.to(torch.int64),
            "lang_masks": lang_masks,
        }
        outs = self._prefill_runner.run(inputs)
        return outs["past_kv"], outs["prefix_pad"]

    def diffusion_step(self, state, prefix_pad, past_kv, x_t, timestep):
        """One flow-matching velocity evaluation."""
        inputs = {
            "state": state.to(torch.float32),
            "x_t": x_t.to(torch.float32),
            "timestep": timestep.to(torch.float32),
            "past_kv": past_kv.to(torch.bfloat16),
            "prefix_pad": prefix_pad,
        }
        outs = self._diffusion_runner.run(inputs)
        return outs["v_t"]

    def diffusion_step_and_vjp(self, state, prefix_pad, past_kv, x_t, timestep, error):
        """RTC F3: returns (v, ∂v/∂x_t @ error). Runs two engines back-to-back."""
        if self._diffusion_bwd_runner is None:
            raise NotSupportedError("RTC bwd engine unavailable")
        v = self.diffusion_step(state, prefix_pad, past_kv, x_t, timestep)
        bwd_inputs = {
            "state": state.to(torch.float32),
            "x_t": x_t.to(torch.float32),
            "timestep": timestep.to(torch.float32),
            "past_kv": past_kv.to(torch.bfloat16),
            "prefix_pad": prefix_pad,
            "error": error.to(torch.float32),
        }
        outs = self._diffusion_bwd_runner.run(bwd_inputs)
        return v, outs["grad_x"]

    # ----- Diagnostic sanity check -----

    @torch.no_grad()
    def _sanity_check(self):
        """Compare TRT vs torch on a single random observation. Logs max_abs."""
        log.info("[trt] running startup sanity check (torch vs TRT single step)...")
        flow = self.flow
        n_act = self.n_act
        C, H, W = 3, 224, 224
        L = flow.config.tokenizer_max_length
        img_head = torch.rand(1, C, H, W, device=self.device)
        img_right = torch.rand(1, C, H, W, device=self.device)
        images = [img_head * 2.0 - 1.0, img_right * 2.0 - 1.0]  # match prepare_images output
        img_masks = [torch.ones(1, dtype=torch.bool, device=self.device),
                     torch.ones(1, dtype=torch.bool, device=self.device)]
        state = torch.randn(1, self.max_state_dim, device=self.device)
        lang_tokens = torch.zeros(1, L, dtype=torch.int64, device=self.device)
        lang_masks = torch.ones(1, L, dtype=torch.bool, device=self.device)

        # torch prefill
        from lerobot.common.policies.pi0.modeling_pi0 import make_att_2d_masks
        prefix_embs, prefix_pad, prefix_att = flow.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        pos_ids = torch.cumsum(prefix_pad, dim=1) - 1
        _, torch_kv = flow.paligemma_with_expert.forward(
            attention_mask=att_2d, position_ids=pos_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True, fill_kv_cache=True,
        )
        torch_kv_stack = torch.stack(
            [torch.stack([torch_kv[i]["key_states"], torch_kv[i]["value_states"]], dim=0)
             for i in range(self.n_layers)], dim=0,
        )

        # TRT prefill
        trt_kv, trt_prefix_pad = self.vlm_prefill(images, img_masks, lang_tokens, lang_masks)

        def _diff(name, a, b):
            d = (a.to(torch.float32) - b.to(torch.float32)).abs()
            print(f"[trt sanity]  {name:<24s} shape={tuple(a.shape)}  max_abs={d.max().item():.3e}  mean_abs={d.mean().item():.3e}", flush=True)
        _diff("past_kv", trt_kv, torch_kv_stack)
        # Layer-level breakdown to find where TRT diverges
        for i in range(min(4, self.n_layers)):
            _diff(f"past_kv[L{i}][K]", trt_kv[i, 0], torch_kv_stack[i, 0])
            _diff(f"past_kv[L{i}][V]", trt_kv[i, 1], torch_kv_stack[i, 1])
        print(f"[trt sanity]  trt_kv range=[{trt_kv.min().item():.3f}, {trt_kv.max().item():.3f}]  "
              f"torch_kv range=[{torch_kv_stack.min().item():.3f}, {torch_kv_stack.max().item():.3f}]", flush=True)
        # Check if TRT L0 K matches any torch Li K (permutation hypothesis)
        trt_l0_k = trt_kv[0, 0].to(torch.float32).flatten()
        best_i, best_diff = -1, float("inf")
        for i in range(self.n_layers):
            d = (trt_l0_k - torch_kv_stack[i, 0].to(torch.float32).flatten()).abs().mean().item()
            if d < best_diff:
                best_diff, best_i = d, i
        print(f"[trt sanity]  trt L0 K best-match torch layer = {best_i} (mean_abs={best_diff:.3e})", flush=True)
        # Sample element checks — TRT[0][K][0,0,0,:8] vs torch[0][K][0,0,0,:8]
        print(f"[trt sanity]  trt L0 K [0,0,0,:8] = {trt_kv[0,0,0,0,0,:8].tolist()}", flush=True)
        print(f"[trt sanity]  torch L0 K [0,0,0,:8] = {torch_kv_stack[0,0,0,0,0,:8].tolist()}", flush=True)
        # Along image tokens: t=0 (first image patch) vs t=256 (last of head img)
        print(f"[trt sanity]  trt L0 K [t=0,:4] = {trt_kv[0,0,0,0,0,:4].tolist()}, "
              f"trt L0 K [t=100,:4] = {trt_kv[0,0,0,100,0,:4].tolist()}", flush=True)
        print(f"[trt sanity]  torch L0 K [t=0,:4] = {torch_kv_stack[0,0,0,0,0,:4].tolist()}, "
              f"torch L0 K [t=100,:4] = {torch_kv_stack[0,0,0,100,0,:4].tolist()}", flush=True)

        # single denoise step
        x_t = torch.randn(1, n_act, self.max_action_dim, device=self.device)
        timestep = torch.ones(1, n_act, device=self.device)
        torch_v = flow._denoise_step_base(state, prefix_pad, torch_kv, x_t, timestep)
        # (a) TRT diffusion with torch KV: isolates diffusion-engine error
        trt_v = self.diffusion_step(state, prefix_pad, torch_kv_stack, x_t, timestep)
        _diff("v_t [torch KV]", trt_v, torch_v)
        # (b) torch diffusion with TRT KV: shows how prefill-engine error propagates
        trt_kv_dict = {
            i: {
                "key_states": trt_kv[i, 0].contiguous(),
                "value_states": trt_kv[i, 1].contiguous(),
            }
            for i in range(self.n_layers)
        }
        v_from_trt_kv = flow._denoise_step_base(state, trt_prefix_pad, trt_kv_dict, x_t, timestep)
        _diff("v_t [TRT KV]", v_from_trt_kv, torch_v)
