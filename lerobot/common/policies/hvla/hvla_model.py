import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel
from transformers.utils import auto_docstring, is_torchdynamo_compiling
from lerobot.common.policies.hvla.action_expert import TransformerFlowMatching, QueryTransformer, CausalTransformer
from lerobot.common.policies.hvla.configuration_hvla import HVLAConfig


class HVLA(Qwen2_5_VLModel):
    r"""Qwen2.5-VL enhanced with learnable K-query tokens.

    Args:
        config:          Qwen2.5-VL configuration.
        num_query:       Number of learnable query tokens to insert.
        freeze_backbone: If True, freeze the vision and language backbone
                         (no gradient updates, switch to eval mode).
    """

    def __init__(self, config: Qwen2_5_VLConfig, vla_config: HVLAConfig):
        super().__init__(config)
        self.action_expert = None
        self.feature_extractor = None
        self.vla_config = vla_config

        freeze_backbone = vla_config.freeze_backbone
        self.latent_dim = vla_config.embed_dim
        self.command_dim = vla_config.command_dim

        # Optionally freeze the backbone
        if freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False
            for p in self.language_model.parameters():
                p.requires_grad = False
            self.visual.eval()
            self.language_model.eval()

    def init_action_expert(self):
        """Initialize the action expert with the given configuration."""
        self.vla_config.q_dim = int(2 * self.vla_config.latent_dim)
        self.vla_config.kv_dim = self.language_model.config.hidden_size
        self.feature_extractor = QueryTransformer(self.vla_config, self.latent_dim * 2)
        self.feature_extractor.to(self.device, dtype=torch.float32)

        self.command_predictor = nn.Linear(self.latent_dim, self.command_dim)
        self.command_predictor.to(self.device, dtype=torch.float32)

        input_dim = self.latent_dim + self.vla_config.obs_dim
        self.action_expert = CausalTransformer(self.vla_config, input_dim=input_dim, output_dim=self.vla_config.act_dim)
        self.action_expert.to(self.device, dtype=torch.float32)

    def save_action_expert(self, save_path):
        state_dict = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'command_predictor': self.command_predictor.state_dict(),
            'action_expert': self.action_expert.state_dict()
        }
        torch.save(state_dict, os.path.join(save_path, "action_expert.pth"))

    def load_action_expert(self, load_path, strict=True):
        state_dict = torch.load(load_path, map_location="cpu")
        self.feature_extractor.load_state_dict(state_dict["feature_extractor"], strict=strict)
        self.action_expert.load_state_dict(state_dict["action_expert"], strict=strict)
        self.command_predictor.load_state_dict(state_dict["command_predictor"], strict=strict)
        print(f"Load pretrained action expert from {load_path}")

    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0)
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                    (input_ids is not None and input_ids.shape[1] != 1)
                    or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                    (cache_position is not None and cache_position[0] == 0)
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        latent_params = self.feature_extractor(last_hidden_state.to(dtype=torch.float32))
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_embeds = mu + std * eps if self.training else mu
        pred_cmds = self.command_predictor(mu)

        return latent_embeds, pred_cmds

    def get_action_loss(self, robot_observations: torch.Tensor, target_action: torch.Tensor):

        robot_observations = robot_observations.to(self.device, dtype=torch.float32)
        loss = F.mse_loss(self.action_expert(robot_observations), target_action)

        return loss

    def pred_action(self, robot_observations: torch.Tensor):
        robot_observations = robot_observations.to(self.device, dtype=torch.float32)
        pred_actions = self.action_expert(robot_observations)

        return pred_actions