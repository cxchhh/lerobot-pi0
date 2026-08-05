#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pi0.5 — a port of openpi's `PI0Pytorch` (pi05=True) onto this fork's policy API.

What differs from pi0, and why the weights are not interchangeable:

  1. **State enters through the prompt.** Instead of a `state_proj` linear layer
     feeding the action expert, the normalized state is discretized into 256 bins
     and rendered into the text prompt:
     `"Task: <task>, State: <b0> <b1> ... <b31>;\\nAction: "`. So there is no
     `state_proj` parameter, and `tokenizer_max_length` is 200 rather than 48.

  2. **The timestep modulates the expert via AdaRMS.** pi0 concatenates the time
     embedding with the action embedding and mixes them with an MLP
     (`action_time_mlp_*`). pi0.5 instead feeds a time embedding as the
     conditioning vector of every expert norm (`time_mlp_*` -> `adarms_cond`),
     see `AdaRMSNorm` in `paligemma_with_expert.py`.

  3. **Quantile normalization.** State and actions use q01/q99 rather than
     mean/std.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import logging
import math
from collections import deque
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.common.constants import ACTION, OBS_STATE
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
    get_gemma_variant,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy

T = TypeVar("T", bound="PI05Policy")

# State is discretized into this many bins before being written into the prompt,
# see openpi `PaligemmaTokenizer.tokenize()`.
N_STATE_BINS = 256


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """float64 is unsupported on MPS; fall back to float32 there."""
    device_type = device if isinstance(device, str) else device.type
    if device_type == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Sine-cosine positional embedding for scalar positions, see openpi."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    device = torch.device(device) if isinstance(device, str) else device

    dtype = get_safe_dtype(torch.float64, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs, so `att_masks` int[B, N] selects the attention
    pattern: all-zeros is full bidirectional attention, a 1 opens a new causal
    block.

    Args:
      pad_masks: bool[B, N] true if part of the input, false if padding.
      att_masks: int32[B, N] 1 where previous tokens cannot depend on it.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Right-pad the last dimension with zeros up to `new_dim`."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad(img, width, height, pad_value=-1):
    """Resize preserving aspect ratio, padding the remainder with `pad_value`.

    Expects [..., C, H, W]. Callers pass images already scaled to [-1, 1], so
    the default padding of -1 is "black" for SigLIP, matching openpi where the
    letterbox is applied to uint8 zeros before normalization.
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # Pad on both sides so the image stays centered, see openpi `resize_with_pad`.
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    return F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)


class PI05FlowMatching(nn.Module):
    """Core pi0.5 model, see openpi `PI0Pytorch`.

    Attribute names are load-bearing: they define the parameter tree that the
    official checkpoints are stored against.
    """

    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config

        expert_width = get_gemma_variant(config.action_expert_variant)["width"]

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_with_export_config, precision=config.dtype
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.max_action_dim)

        # Timestep -> AdaRMS conditioning vector. Note there is no action/time
        # fusion MLP here (unlike pi0): the time signal reaches the expert only
        # through the norms.
        self.time_mlp_in = nn.Linear(expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)

        # Set by `forward` when training-time RTC is on, read by the policy to
        # exclude prefix positions from the loss denominator.
        self.last_suffix_mask = None

        self.set_requires_grad()

    def set_requires_grad(self):
        # `train_expert_only` / `freeze_vision_encoder` only ever freeze parts of
        # the VLM tower. The action expert and the action/time projections are
        # what adapt the policy to a new robot, so they always stay trainable.
        self.paligemma_with_expert.set_requires_grad()

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images with SigLIP and language (task + discretized state) tokens.

        Unlike pi0 there is no state embedding here: the state already rode in
        through `lang_tokens`.
        """
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Full bidirectional attention between image and language tokens.
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed the noisy action chunk, and build the AdaRMS conditioning from `timestep`.

        `timestep` is [B] for ordinary flow matching, or [B, T] when
        training-time RTC drives prefix positions at t=0 and the rest at t.
        """
        per_position = timestep.ndim == 2
        time_emb = create_sinusoidal_pos_embedding(
            timestep.reshape(-1),
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        if per_position:
            time_emb = time_emb.reshape(*timestep.shape, -1)
        time_emb = time_emb.type(dtype=noisy_actions.dtype)

        action_emb = self.action_in_proj(noisy_actions)

        x = self.time_mlp_in(time_emb)
        x = F.silu(x)  # swish == silu
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)

        bsize, action_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_dim, dtype=torch.bool, device=action_emb.device)

        # Image, language and state tokens must not attend to action tokens.
        att_masks = [1] + ([0] * (self.config.chunk_size - 1))
        att_masks = torch.tensor(att_masks, dtype=action_emb.dtype, device=action_emb.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return action_emb, pad_masks, att_masks, adarms_cond

    def random_prefix_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time RTC prefix sampler (arXiv:2512.05964 §3).

        Identical sampling to `PI0FlowMatching.random_prefix_mask`, so the two
        policies see the same objective:
          1) With probability `train_time_rtc_prefix_drop_p`, FORCE d=0.
          2) Otherwise sample d ~ U(0, floor(T * max_prefix_frac) + 1).

        Positions [0, d) are the prefix (mask=0, already-committed clean
        actions); [d, T) is the suffix (mask=1, trained under flow matching).
        """
        B, T, _ = x.shape
        device = x.device
        max_frac = float(self.config.train_time_rtc_max_prefix_frac)
        drop_p = float(self.config.train_time_rtc_prefix_drop_p)
        max_prefix = max(0, min(T - 1, int(T * max_frac)))
        prefix_lens = torch.randint(low=0, high=max_prefix + 1, size=(B,), device=device)
        if drop_p > 0.0:
            drop_mask = torch.rand(B, device=device) < drop_p
            prefix_lens = torch.where(drop_mask, torch.zeros_like(prefix_lens), prefix_lens)
        t_idx = torch.arange(T, device=device)[None, :]
        mask_suffix = (t_idx >= prefix_lens[:, None]).float()
        return mask_suffix.unsqueeze(-1)

    def forward(self, images, img_masks, lang_tokens, lang_masks, actions, noise=None, time=None) -> Tensor:
        """Training forward pass; returns the per-element flow-matching loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        B, T, _ = actions.shape
        time_expanded = time[:, None, None].expand(-1, T, 1)
        # Training-time RTC: mask[t]=0 marks a prefix position -> x_t there is the
        # clean action, u_t=0 (no loss), and the AdaRMS conditioning sees t=0.
        if self.config.train_time_rtc:
            time_prefix_mask = self.random_prefix_mask(time_expanded)
        else:
            time_prefix_mask = torch.ones_like(time_expanded)
        # Exposed so the outer policy.forward can drop prefix positions from the
        # mean-loss denominator.
        self.last_suffix_mask = time_prefix_mask.detach()

        time_masked = time_expanded * time_prefix_mask  # (B, T, 1)
        x_t = time_masked * noise + (1.0 - time_masked) * actions
        u_t = (noise - actions) * time_prefix_mask

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time_masked.squeeze(-1)
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        # Zero out the loss on prefix positions (training-time RTC).
        return losses * time_prefix_mask

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, noise=None, action_prefix=None
    ) -> Tensor:
        """Integrate the learned flow from noise (t=1) to actions (t=0).

        `action_prefix` [B, d, D] are already-committed actions. Because the
        model was trained with RTC, they are simply pinned into `x_t[:, :d]` and
        driven at t=0 — no ΠGDM inpainting needed at inference.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_len = 0 if action_prefix is None else action_prefix.shape[1]
        if prefix_len > self.config.chunk_size:
            raise ValueError(
                f"`action_prefix` is longer ({prefix_len}) than the chunk ({self.config.chunk_size})."
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Fill the KV cache once; the prefix is fixed across denoising steps.
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_inference_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise.clone()
        if prefix_len:
            x_t[:, :prefix_len] = action_prefix.to(dtype=x_t.dtype)

        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            if prefix_len:
                # Prefix positions are already clean, so they ride at t=0.
                expanded_time = time.expand(bsize, self.config.chunk_size).clone()
                expanded_time[:, :prefix_len] = 0.0
            else:
                expanded_time = time.expand(bsize)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, expanded_time)

            # Euler step
            x_t = x_t + dt * v_t
            if prefix_len:
                # Keep the committed actions pinned against integration drift.
                x_t[:, :prefix_len] = action_prefix.to(dtype=x_t.dtype)
            time += dt
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """Apply one denoising step to `x_t` at the given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """Wrapper around `PI05FlowMatching` to train and run inference within LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self._check_feature_dims(config)
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224", local_files_only=True
        )
        self.model = PI05FlowMatching(config)

        self.reset()

    @staticmethod
    def _check_feature_dims(config: PI05Config) -> None:
        """Fail early on dimensions the action projections cannot represent."""
        action_ft = config.output_features.get(ACTION)
        if action_ft is not None and action_ft.shape[0] > config.max_action_dim:
            raise ValueError(
                f"The dataset's action dimension ({action_ft.shape[0]}) exceeds `max_action_dim` "
                f"({config.max_action_dim}). Raise `max_action_dim` to at least {action_ft.shape[0]} — "
                "the pretrained action projections are widened automatically, with the extra "
                "dimensions starting from zero."
            )

        state_ft = config.input_features.get(OBS_STATE)
        if state_ft is not None and state_ft.shape[0] > config.max_state_dim:
            # Harmless mechanically (state is text, not a projection), but it makes the
            # prompt longer than anything seen in pretraining.
            logging.warning(
                f"pi05: state dimension ({state_ft.shape[0]}) exceeds `max_state_dim` "
                f"({config.max_state_dim}), so the prompt will carry {state_ft.shape[0]} discretized values "
                f"instead of {config.max_state_dim}. Consider raising `max_state_dim`."
            )

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        """Load a checkpoint, remapping official pi0.5 parameter names if needed.

        The official `lerobot/pi05_*` checkpoints were exported against a newer
        `transformers` PaliGemma layout (`paligemma.model.language_model.*`, with
        `embed_tokens` omitted because it is tied to `lm_head`) and without the
        `model.` prefix this fork's policy adds, so `--policy.path=lerobot/pi05_base`
        needs `_remap_official_state_dict` to line the two up.
        """
        from safetensors.torch import load_file

        state_dict = load_file(model_file, device=map_location)
        state_dict = cls._remap_official_state_dict(state_dict)
        state_dict = cls._adapt_action_dim(state_dict, model)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # The official checkpoints carry no normalization buffers: those stats
        # come from the finetuning dataset. Anything else missing is a real error.
        norm_missing = [k for k in missing if k.startswith(("normalize_", "unnormalize_"))]
        missing = [k for k in missing if k not in norm_missing]
        if norm_missing:
            logging.info(
                f"pi05 checkpoint has no normalization stats ({len(norm_missing)} buffers); they must come "
                "from `dataset_stats` or a later `load_state_dict`."
            )
        if missing or unexpected:
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n"
                f"  Missing keys: {missing[:10]}\n  Unexpected keys: {unexpected[:10]}"
            )

        return model

    @staticmethod
    def _remap_official_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Translate official pi05 checkpoint keys into this fork's parameter tree."""
        if any(k.startswith("model.paligemma_with_expert.") for k in state_dict):
            # Already in this fork's layout (a checkpoint we saved ourselves).
            remapped = dict(state_dict)
        else:
            # Newer transformers nests the towers under `paligemma.model.*` and hoists
            # `lm_head`; 4.48 keeps `vision_tower`/`multi_modal_projector` at the top and
            # puts the decoder under `language_model.model.*`.
            pg = "paligemma_with_expert.paligemma."
            renames = (
                (f"{pg}model.language_model.", f"{pg}language_model.model."),
                (f"{pg}model.vision_tower.", f"{pg}vision_tower."),
                (f"{pg}model.multi_modal_projector.", f"{pg}multi_modal_projector."),
                (f"{pg}lm_head.", f"{pg}language_model.lm_head."),
            )

            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                for old, new in renames:
                    if new_key.startswith(old):
                        new_key = new_key.replace(old, new, 1)
                        break
                remapped[f"model.{new_key}"] = value

        # `embed_tokens` is tied to `lm_head`, so it is absent from both the
        # official checkpoints and the ones we save (safetensors drops tied
        # duplicates), yet it is a distinct state_dict entry in this
        # transformers version.
        lm_head = "model.paligemma_with_expert.paligemma.language_model.lm_head.weight"
        embed_tokens = "model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight"
        if lm_head in remapped and embed_tokens not in remapped:
            remapped[embed_tokens] = remapped[lm_head]

        return remapped

    @staticmethod
    def _adapt_action_dim(state_dict: dict[str, Tensor], model: "PI05Policy") -> dict[str, Tensor]:
        """Grow the action projections when finetuning at a larger `max_action_dim`.

        The released checkpoints are trained at `max_action_dim=32`, the width of
        pi0's shared action space. A robot with more actuated dimensions needs
        wider `action_in_proj` / `action_out_proj`, so the pretrained block is
        copied in and the extra rows/columns start at zero: the first 32
        dimensions behave exactly as pretrained, the rest are learned from
        scratch during finetuning.
        """
        adapted = dict(state_dict)
        for name in ("action_in_proj", "action_out_proj"):
            for suffix in ("weight", "bias"):
                key = f"model.{name}.{suffix}"
                if key not in adapted:
                    continue
                pretrained = adapted[key]
                target = model.state_dict()[key]
                if pretrained.shape == target.shape:
                    continue
                if any(t < p for t, p in zip(target.shape, pretrained.shape, strict=True)):
                    raise ValueError(
                        f"Cannot load '{key}': checkpoint has shape {tuple(pretrained.shape)} but the model "
                        f"expects {tuple(target.shape)}. `max_action_dim` can be raised above the "
                        f"checkpoint's value but not lowered."
                    )
                grown = torch.zeros_like(target)
                grown[tuple(slice(0, s) for s in pretrained.shape)] = pretrained
                adapted[key] = grown
                logging.warning(
                    f"pi05: widened {key} from {tuple(pretrained.shape)} to {tuple(target.shape)} for the "
                    "larger `max_action_dim`; the added dimensions are zero-initialized and must be finetuned."
                )
        return adapted

    def _pad_and_discretize_state(self, state: Tensor) -> np.ndarray:
        """Pad the normalized state to `max_state_dim` and bin it into [0, 255].

        See openpi `PaligemmaTokenizer.tokenize()`. The state is expected to be
        in [-1, 1] already (quantile normalization ran upstream).
        """
        state = pad_vector(state, self.config.max_state_dim)
        bins = np.linspace(-1, 1, N_STATE_BINS + 1)[:-1]
        return np.digitize(state.float().cpu().numpy(), bins=bins) - 1

    @torch.no_grad
    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Build and tokenize the pi0.5 prompt: task text plus the discretized state.

        With `n_obs_states > 1` the state arrives as [B, n_obs_states, D] (oldest
        first, current last), matching the state history pi0 feeds as extra
        expert tokens. The older frames are written as a `History:` clause and
        the current frame keeps the exact `State: ...;\\nAction: ` tail that
        pi0.5 was pretrained with.
        """
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        discretized_states = self._pad_and_discretize_state(batch[OBS_STATE])
        if discretized_states.ndim == 2:  # [B, D] -> [B, 1, D]
            discretized_states = discretized_states[:, None, :]

        prompts = []
        for task, frames in zip(tasks, discretized_states, strict=True):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, frames[-1]))
            if len(frames) > 1:
                history_str = " | ".join(" ".join(map(str, f)) for f in frames[:-1])
                prompts.append(
                    f"Task: {cleaned_text}, History: {history_str}, State: {state_str};\nAction: "
                )
            else:
                prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

        tokenized_prompt = self.language_tokenizer.__call__(
            prompts,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_images(self, batch):
        """Apply SigLIP preprocessing: [0, 1] -> [-1, 1], letterboxed to the model resolution.

        Cameras declared in the config but absent from the batch are fed as
        all-black images with a zeroed mask.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # Normalize first so the letterbox padding (-1) is black in SigLIP space.
            img = img * 2.0 - 1.0
            if img.shape[-2:] != tuple(self.config.image_resolution):
                img = resize_with_pad(img, *self.config.image_resolution, pad_value=-1)

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad the state to `max_state_dim`."""
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch):
        """Pad the action chunk to `max_action_dim`."""
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Actions are produced in chunks of `chunk_size`; the first
        `n_action_steps` are queued and returned one at a time.
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # `self._action_queue` has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a full chunk of actions given environment observations."""
        self.eval()

        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks)

        # Unpad the actions back to the environment's action dimension.
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return self.unnormalize_outputs({ACTION: actions})[ACTION]

    @torch.no_grad
    def get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return a full action chunk for the deployment stack (see `server.py`).

        Same contract as `PI0Policy.get_action_chunk`: takes the raw observation
        dict, returns `(chunk_size, action_dim)` in the environment's action
        space (batch dimension squeezed out).

        Optional keys, all forwarded by `server.py` from the client:
          - `action_prefix` [1, d, D]: actions already committed by the previous
            chunk, in *unnormalized* space.
          - `delay`: the paper's inference delay d — how many of those actions
            will have been executed by the time this chunk lands. Only the first
            `delay` entries are pinned.
          - `rtc_prefix_attention_horizon`, `rtc_max_guidance_weight`: accepted
            and ignored. They tune pi0's inference-time ΠGDM soft guidance;
            pi0.5 was trained with training-time RTC, whose whole point is that
            the prefix is simply held fixed at t=0 with no guidance term.
        """
        self.eval()

        delay = 0
        if "delay" in batch:
            delay = int(batch["delay"].reshape(-1)[0].item())

        # The prefix must be in the same normalized space as the model's noise,
        # so route it through `normalize_targets` alongside the action target.
        has_prefix = "action_prefix" in batch
        if has_prefix:
            batch = dict(batch)
            batch[ACTION] = batch["action_prefix"]
        batch = self.normalize_inputs(batch)
        if has_prefix:
            batch = self.normalize_targets(batch)
            batch["action_prefix"] = batch[ACTION]

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        action_prefix = None
        if has_prefix:
            action_prefix = self.prepare_action(batch)
            # Pin only what will actually have been executed; `delay=0` (or a
            # missing key) means honor the whole prefix the client sent.
            prefix_len = delay if delay > 0 else action_prefix.shape[1]
            prefix_len = min(prefix_len, action_prefix.shape[1], self.config.chunk_size)
            action_prefix = action_prefix[:, :prefix_len] if prefix_len else None

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, noise=None, action_prefix=action_prefix
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions.squeeze(0)

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the batch through the model and compute the loss for training."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        # Note the singular "action": that is the key `LeRobotDataset` derives from
        # the `action` entry of `delta_timestamps`. Getting this name wrong silently
        # trains on the frames used to pad a chunk past the end of an episode.
        action_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        # Suffix mask (1 = suffix / trained; 0 = training-time RTC prefix).
        # Losses are already zeroed on prefix by model.forward; excluding those
        # positions from the denominator too keeps the effective per-position
        # gradient invariant to the sampled prefix length.
        suffix_mask = self.model.last_suffix_mask
        if suffix_mask is None:
            suffix_mask = torch.ones_like(losses[..., :1])

        if action_is_pad is not None:
            in_episode_bound = ~action_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()
            valid_mask = suffix_mask.squeeze(-1) * in_episode_bound.float()
        else:
            valid_mask = suffix_mask.squeeze(-1)

        # Remove the padding dimensions added by `prepare_action`.
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict["losses_after_rm_padding"] = losses.clone()

        # Denominator counts only non-pad suffix positions x action_dim.
        den = valid_mask.sum().clamp_min(1) * losses.shape[-1]
        loss = losses.sum() / den
        # For backward compatibility, the mean of the losses is logged as `l2_loss`.
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict
