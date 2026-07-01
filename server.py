from typing import Dict
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

def process_img(img):
    img = torch.from_numpy(img)
    if img.ndim == 3 or img.ndim == 4 and img.shape[0] != 1:
        img = img.unsqueeze(0)
    img = einops.rearrange(img, "... h w c -> ... c h w").contiguous()
    img = img.type(torch.float32)
    img /= 255
    return img

class ServerPolicy(_base_policy.BasePolicy):
    def __init__(self, model: PreTrainedPolicy, device: str, save_attn: bool = False):
        self.model = model
        self.device = device
        self.save_attn = save_attn
        self._expert = model.model.paligemma_with_expert
        self._expert._save_attn = save_attn
        self._attn_step = 0

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
        qpos = obs_dict['observation.state']
        observation = dict()
        observation['observation.state'] = torch.tensor(np.array(qpos)).unsqueeze(0).float().to(self.device)
        for key in obs_dict:
            if "images" in key:
                observation[key] = process_img(np.array(obs_dict[key])).to(self.device)
        observation['task'] = [obs_dict['task']]
        observation["task_index"] = torch.tensor(0).unsqueeze(0).to(self.device)
        if "action_prefix" in obs_dict.keys():
            observation["action_prefix"] = torch.tensor(np.array(obs_dict['action_prefix'])).unsqueeze(0).float().to(self.device)
        if "delay" in obs_dict.keys():
            observation["delay"] = torch.tensor(obs_dict['delay']).unsqueeze(0).to(self.device)
        if "rtc_prefix_attention_horizon" in obs_dict.keys():
            observation["rtc_prefix_attention_horizon"] = torch.tensor(
                obs_dict["rtc_prefix_attention_horizon"]).unsqueeze(0).to(self.device)
        if "rtc_max_guidance_weight" in obs_dict.keys():
            observation["rtc_max_guidance_weight"] = torch.tensor(
                obs_dict["rtc_max_guidance_weight"], dtype=torch.float32).unsqueeze(0).to(self.device)
            
        if obs_dict['reset']:
            self.model.reset()

        try:
            action = self.model.get_action_chunk(observation).cpu().numpy()
        except Exception:
            import traceback as _tb
            print("=" * 60, flush=True)
            print("[server] get_action_chunk failed; full traceback:", flush=True)
            _tb.print_exc()
            print("=" * 60, flush=True)
            raise
        if self.save_attn:
            self._visualize_attention(obs_dict)
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

    
    save_attn = os.environ.get("SAVE_ATTN", "0") == "1"
    policy = ServerPolicy(model=network, device=cfg.policy.device, save_attn=save_attn)
    policy_server = WebsocketPolicyServer(policy=policy, host=HOST, port=PORT)
    print(f"Starting server on {HOST}:{PORT}")
    try:
        policy_server.serve_forever()
    finally:
        if save_attn:
            cv2.destroyAllWindows()
    

if __name__ == '__main__':
    init_logging()
    main_wrapper()