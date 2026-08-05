from typing import Dict
import sys
import einops
from termcolor import colored
import torch
import os
import numpy as np
from pathlib import Path
import json
import logging
import cv2
from dataclasses import asdict
from pprint import pformat
from gx_infer import base_policy as _base_policy
from gx_infer.websocket_policy_server import WebsocketPolicyServer

from lerobot.common.datasets.factory import IMAGENET_STATS
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

HOST = "0.0.0.0"
PORT = "8002"


def _pop_bool(argv: list, name: str) -> bool:
    if name in argv:
        argv.remove(name)
        return True
    return False


def _pop_kv(argv: list, name: str, default=None):
    """Remove `--name value` or `--name=value` from argv, return the value.
    Leaves argv untouched (returning default) if the flag isn't present.
    Needed because ``parser.wrap()`` (draccus-backed) would reject unknown args.
    """
    for i, arg in enumerate(argv):
        if arg == name:
            argv.pop(i)
            if i < len(argv):
                return argv.pop(i)
            return ""
        if arg.startswith(name + "="):
            argv.pop(i)
            return arg[len(name) + 1:]
    return default


# --- TRT CLI flags (pre-parsed before draccus so it doesn't choke on them) ---
_TRT_ENABLED = _pop_bool(sys.argv, "--trt")
_TRT_PRECISION = _pop_kv(sys.argv, "--trt-precision", "bf16")
_TRT_RTC_MODE = _pop_kv(sys.argv, "--trt-rtc-mode", "torch")
_TRT_CACHE_DIR = _pop_kv(sys.argv, "--trt-cache-dir", None)
_TRT_DEBUG = _pop_bool(sys.argv, "--trt-debug")
# --trt-mode picks which engine to actually use once TRT is on:
#   diff-only  (default) — vlm stays in torch (bit-exact KV), diffusion on TRT.
#                          Best precision, only ~7ms slower than full.
#   full       — both vlm and diffusion on TRT. Fastest, but vlm KV bf16 kernel
#                differences propagate through attention and hurt precision.
#   vlm-only   — vlm on TRT, diffusion in torch. Rarely useful (vlm alone is
#                <5ms, barely any speedup).
_TRT_MODE = _pop_kv(sys.argv, "--trt-mode", "diff-only")

def process_img(img, device="cpu"):
    # Upload the uint8 tensor first, then convert on-device: the H2D copy is
    # 1/4 the bytes of shipping float32 across PCIe.
    img = torch.from_numpy(img).to(device, non_blocking=True)
    if img.ndim == 3 or img.ndim == 4 and img.shape[0] != 1:
        img = img.unsqueeze(0)
    img = einops.rearrange(img, "... h w c -> ... c h w").contiguous()
    img = img.type(torch.float32)
    # 0-dim tensor divisor: CUDA's tensor/scalar path multiplies by the
    # reciprocal (1 ULP off vs CPU `/ 255`); tensor/tensor stays bit-exact.
    img /= img.new_full((), 255.0)
    return img

class ServerPolicy(_base_policy.BasePolicy):
    def __init__(self, model: PreTrainedPolicy, device: str, save_attn: bool = False):
        self.model = model
        self.device = device
        self.save_attn = save_attn
        self._expert = model.model.paligemma_with_expert
        self._expert._save_attn = save_attn
        self._attn_step = 0
        # Rolling window of end-to-end infer() wall time (ms). Every 20 calls
        # we print mean / p50 / p95 so a running rollout shows the current
        # latency without needing to kill the server.
        from collections import deque as _deque
        self._infer_times = _deque(maxlen=200)

    def _visualize_attention(self, obs_dict):
        attn = getattr(self._expert, '_attn_probs', None)
        if attn is None:
            return
        n_act = self.model.config.n_action_steps
        patch_size = 14
        grid_size = 16  # 224 / 14
        num_img_tokens = grid_size * grid_size

        overlays = []
        for cam_idx, cam_key in enumerate(["observation.images.head", "observation.images.right_wrist"]):
            if cam_key not in obs_dict:
                continue
            img = np.array(obs_dict[cam_key])
            if img.ndim == 4:
                img = img[0]
            orig_h, orig_w = img.shape[:2]

            # 计算 resize_with_pad 后的有效区域
            target_size = 224
            ratio = max(orig_w / target_size, orig_h / target_size)
            resized_h = int(orig_h / ratio)
            resized_w = int(orig_w / ratio)
            valid_patch_h = resized_h // patch_size
            valid_patch_w = resized_w // patch_size

            start = cam_idx * num_img_tokens
            end = start + num_img_tokens
            cam_attn = attn[0, :, -n_act:, start:end].mean(dim=(0, 1)).numpy()
            cam_attn = cam_attn.reshape(grid_size, grid_size)
            # 只取有效 patch 区域
            cam_attn = cam_attn[:valid_patch_h, :valid_patch_w]
            # 去掉边缘 1 圈 patch 的影响（ViT attention sink）
            if valid_patch_h > 2 and valid_patch_w > 2:
                inner = cam_attn[1:-1, 1:-1]
                edge_val = inner.mean()
                cam_attn[0, :] = edge_val
                cam_attn[-1, :] = edge_val
                cam_attn[:, 0] = edge_val
                cam_attn[:, -1] = edge_val
            cam_attn = (cam_attn - cam_attn.min()) / (cam_attn.max() - cam_attn.min() + 1e-8)

            attn_resized = cv2.resize(cam_attn, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
            overlays.append(overlay)

        if overlays:
            combined = np.hstack(overlays)
            cv2.imshow("attention", combined)
            cv2.waitKey(1)

        self._expert._attn_probs = None
        self._expert._attn_probs_sum = None
        self._expert._attn_weight_sum = 0
        self._expert._attn_layer_idx = 0

    def infer(self, obs_dict: Dict) -> Dict:
        import time as _time
        _t0 = _time.perf_counter()

        qpos = obs_dict['observation.state']
        observation = dict()
        observation['observation.state'] = torch.tensor(np.array(qpos)).unsqueeze(0).float().to(self.device)
        for key in obs_dict:
            if "images" in key:
                observation[key] = process_img(np.array(obs_dict[key]), self.device)
        observation['task'] = [obs_dict['task']]
        observation["task_index"] = torch.tensor(0).unsqueeze(0).to(self.device)
        if "action_prefix" in obs_dict.keys():
            observation["action_prefix"] = torch.tensor(np.array(obs_dict['action_prefix'])).unsqueeze(0).float().to(self.device)
        # Scalar knobs stay on CPU. They're only consumed via `.item()` inside
        # get_action_chunk, and pushing them to GPU forces a stream sync each
        # time (~5-10 ms per `.item()` on cuda tensors). Keeping them CPU makes
        # `.item()` a cheap python read.
        if "delay" in obs_dict.keys():
            observation["delay"] = torch.tensor(obs_dict['delay']).unsqueeze(0)
        if "rtc_prefix_attention_horizon" in obs_dict.keys():
            observation["rtc_prefix_attention_horizon"] = torch.tensor(
                obs_dict["rtc_prefix_attention_horizon"]).unsqueeze(0)
        if "rtc_max_guidance_weight" in obs_dict.keys():
            observation["rtc_max_guidance_weight"] = torch.tensor(
                obs_dict["rtc_max_guidance_weight"], dtype=torch.float32).unsqueeze(0)

        if obs_dict['reset']:
            self.model.reset()

        torch.cuda.synchronize()
        _t_preproc = _time.perf_counter()

        try:
            action_gpu = self.model.get_action_chunk(observation)
        except Exception:
            import traceback as _tb
            print("=" * 60, flush=True)
            print("[server] get_action_chunk failed; full traceback:", flush=True)
            _tb.print_exc()
            print("=" * 60, flush=True)
            raise

        torch.cuda.synchronize()
        _t_model = _time.perf_counter()

        action = action_gpu.cpu().numpy()

        if self.save_attn:
            self._visualize_attention(obs_dict)

        _t_end = _time.perf_counter()
        _dt_ms = (_t_end - _t0) * 1000.0
        preproc_ms = (_t_preproc - _t0) * 1000.0
        model_ms = (_t_model - _t_preproc) * 1000.0
        d2h_ms = (_t_end - _t_model) * 1000.0

        self._infer_times.append((_dt_ms, preproc_ms, model_ms, d2h_ms))
        if len(self._infer_times) % 20 == 0:
            def _stats(vs):
                s = sorted(vs); n = len(s)
                return sum(s) / n, s[n // 2], s[min(n - 1, int(n * 0.95))]
            totals = [x[0] for x in self._infer_times]
            pre = [x[1] for x in self._infer_times]
            mdl = [x[2] for x in self._infer_times]
            d2h = [x[3] for x in self._infer_times]
            n = len(totals)
            m_t, p50_t, p95_t = _stats(totals)
            m_pre, p50_pre, _ = _stats(pre)
            m_mdl, p50_mdl, _ = _stats(mdl)
            m_d2h, p50_d2h, _ = _stats(d2h)
            print(
                f"[infer] n={n} total mean={m_t:5.1f}/p50={p50_t:5.1f}/p95={p95_t:5.1f}ms  "
                f"[preproc {m_pre:4.1f}ms | model {m_mdl:5.1f}ms | d2h {m_d2h:4.1f}ms]",
                flush=True,
            )
        return {"actions": action }

    def on_disconnect(self):
        if self.save_attn:
            cv2.destroyAllWindows()

@parser.wrap()
def main_wrapper(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(asdict(cfg)))
    # Check device is available
    cfg.policy.device = "cuda" if torch.cuda.is_available() else "cpu"
    # configuration
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info("load dataset metainfo")
    ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )

    # load policy
    logging.info("Making policy.")
    if cfg.policy.type == "hvla":
        cfg.policy.eval = True
        cfg.policy.load_path = cfg.policy.pretrained_path

    network: PreTrainedPolicy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta
    )
    network.eval()

    # --- Optional TensorRT backend (--trt) ---
    if _TRT_ENABLED:
        from lerobot.common.policies.pi0.trt_infer import PI0TRTBackend
        ckpt_dir = Path(_TRT_CACHE_DIR) if _TRT_CACHE_DIR else Path(
            cfg.policy.pretrained_path or cfg.policy.path
        )
        logging.info(colored(
            f"[trt] enabling TRT backend: precision={_TRT_PRECISION} "
            f"mode={_TRT_MODE} rtc_mode={_TRT_RTC_MODE} debug={_TRT_DEBUG} cache={ckpt_dir}",
            "cyan", attrs=["bold"],
        ))
        backend = PI0TRTBackend(
            torch_policy=network,
            ckpt_dir=ckpt_dir,
            precision=_TRT_PRECISION,
            rtc_mode=_TRT_RTC_MODE,
            debug=_TRT_DEBUG,
        )
        network.model.trt_backend = backend
        network.model.trt_rtc_mode = _TRT_RTC_MODE
        if _TRT_MODE == "diff-only":
            network.model.trt_diffusion_only = True
        elif _TRT_MODE == "vlm-only":
            network.model.trt_vlm_only = True
        elif _TRT_MODE == "full":
            pass
        else:
            raise ValueError(f"Unknown --trt-mode {_TRT_MODE!r}; expected diff-only / full / vlm-only")
        logging.info(colored(f"[trt] backend attached (mode={_TRT_MODE}).", "green"))

    save_attn = os.environ.get("SAVE_ATTN", "0") == "1"
    policy = ServerPolicy(model=network, device=cfg.policy.device, save_attn=save_attn)
    # sonic-latent 消融 ckpt（action=78=token64+双手14）：首帧 metadata 自报
    # 后端身份，locomanip run_sim.sh 的 probe 据此自动换 --tracker sonic +
    # --vla-backend pi0_sonic。42/40 维常规 ckpt 不带 metadata（现状不变）。
    _action_dim = int(cfg.policy.action_feature.shape[0])
    _metadata = None
    if _action_dim == 78:
        _metadata = {"backend": "pi0_sonic", "action_dim": 78,
                     "chunk_size": int(cfg.policy.chunk_size)}
        logging.info(f"[sonic] latent ckpt detected: metadata={_metadata}")
    policy_server = WebsocketPolicyServer(
        policy=policy, host=HOST, port=PORT,
        **({"metadata": _metadata} if _metadata else {}))
    print(f"Starting server on {HOST}:{PORT}")
    try:
        policy_server.serve_forever()
    finally:
        if save_attn:
            cv2.destroyAllWindows()
    

if __name__ == '__main__':
    init_logging()
    main_wrapper()