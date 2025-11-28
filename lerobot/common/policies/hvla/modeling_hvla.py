from contextlib import nullcontext
import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import Qwen2_5_VLConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

from lerobot.common.constants import ACTION, OBS_STATE
from lerobot.common.policies.hvla.hvla_model import HVLA
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.hvla.configuration_hvla import HVLAConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.hvla.utils import build_messages
from einops import rearrange

class HVLAPolicy(PreTrainedPolicy):
    config_class = HVLAConfig
    name = "hvla"

    def __init__(self,
        config: HVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32 # torch.bfloat16 if config.bf16 else torch.float32

        self.proc = AutoProcessor.from_pretrained(config.vlm_model, use_fast=True)
        llm_config = Qwen2_5_VLConfig.from_pretrained(
            config.vlm_model,
            attn_implementation="flash_attention_2" if config.bf16 else "eager",
        )
        self.model = HVLA.from_pretrained(
            pretrained_model_name_or_path=config.vlm_model,
            config=llm_config,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="flash_attention_2" if config.bf16 else "eager",
            vla_config=config
        )
        self.model.init_action_expert()

        if config.load_path != "":
            self.model.load_action_expert(config.load_path, strict=True)

        self.max_seq_len = config.max_seq_len
        self.sample_steps = config.sample_steps
        self.act_list = []

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # campat with compute_loss
        k = self.config.n_obs_steps
        language_cmds = [s for _ in range(k) for s in batch["task"]]
        action_gt = batch["action"] # (B, k, A)
        image_list_head = rearrange(batch["observation.images.head"], "b k c h w -> (b k) c h w")
        image_list_r = rearrange(batch["observation.images.right_wrist"], "b k c h w -> (b k) c h w")
        image_list = [[image_head, image_r] for image_head, image_r in zip(image_list_head, image_list_r)]
        obs = rearrange(batch["observation.state"], "b k d -> (b k) d")
        dummy_cmd_gt = torch.zeros([obs.shape[0], self.config.command_dim], device=self.device, dtype=self.dtype)

        loss, info = self.compute_loss(language_cmds, action_gt, dummy_cmd_gt, obs, image_list)
        return loss, info
        
    def compute_loss(self, language_cmds: list, action_gt: torch.Tensor, cmds_gt: torch.Tensor,
                     obs: torch.Tensor, image_list: list = None):
        messages = [build_messages(language_cmds[i], image_list[i] if image_list else []) for i in
                    range(len(language_cmds))]
        language_cmds = self.proc.apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        vlm_inputs = self.proc(text=language_cmds, images=image_inputs, return_tensors="pt", padding=True)
        vlm_inputs = vlm_inputs.to(device=self.device)

        bad = [n for n,p in self.model.named_parameters() if p.dtype != torch.float32]
        assert len(bad) == 0, f"Found non-fp32 params: {bad[:5]}..."
        
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.config.bf16 else nullcontext():
            latent_embedding, pred_cmds = self.model(**vlm_inputs)
            cmd_loss = torch.nn.functional.mse_loss(pred_cmds, cmds_gt.to(self.device, dtype=torch.bfloat16))
        
        obs = torch.concatenate([obs.float(), latent_embedding.float()], dim=-1) # (B*k, 1, p+D)
        obs = rearrange(obs, "(b k) d -> b k d", k=self.config.n_obs_steps) # (B, k, p+D)
        

        with torch.amp.autocast("cuda", enabled=False) if self.config.bf16 else nullcontext():
            pred_motor_targets = self.model.pred_action(obs)
            action_loss = self.model.get_action_loss(obs, action_gt.float())
            total_loss = action_loss + cmd_loss.float()
            

        info = {"action_loss": action_loss.item(), "cmd_loss":cmd_loss.item(), "action": pred_motor_targets[:, -1, :]}
        return total_loss, info


    @torch.no_grad()
    def infer(self, language_cmds: list, obs: torch.Tensor, image_list: list = None):
        messages = [build_messages(language_cmds[i], image_list[i] if image_list else []) for i in
                    range(len(language_cmds))]
        language_cmds = self.proc.apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        vlm_inputs = self.proc(text=language_cmds, images=image_inputs, return_tensors="pt", padding=True)
        vlm_inputs = vlm_inputs.to(device=self.device, dtype=self.dtype)

        latent_embedding, pred_cmds = self.model(**vlm_inputs)
        pred_cmds_numpy = pred_cmds.float().detach().cpu().numpy()

        obs = torch.concatenate([obs, latent_embedding], dim=-1)
        if self.obs_list:
            self.obs_list[-1].detach_()
        self.obs_list.append(obs)
        self.obs_list = self.obs_list[-self.max_seq_len:]
        obs_list = torch.stack(self.obs_list, dim=1)
        obs_list = obs_list.to(device=self.device, dtype=self.dtype)

        pred_motor_targets = self.model.pred_action(obs_list)

        return pred_motor_targets[:, -1, :], pred_cmds_numpy

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        language_cmds = batch["task"]
        action_gt = batch["action"]
        image_list_head = rearrange(batch["observation.images.head"], "b k c h w -> (b k) c h w")
        image_list_r = rearrange(batch["observation.images.right_wrist"], "b k c h w -> (b k) c h w")
        image_list = [[image_head, image_r] for image_head, image_r in zip(image_list_head, image_list_r)]
        obs = rearrange(batch["observation.state"], "b k d -> (b k) d")
        breakpoint()
        return self.infer(language_cmds, obs, image_list)

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self.act_list = []
        pass