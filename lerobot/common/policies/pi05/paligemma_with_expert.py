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

"""PaliGemma VLM + Gemma action expert, wired the way openpi's pi0.5 does it.

Self-contained on purpose: the layer loop, RoPE and attention are implemented
here rather than delegated to `transformers`, because pi0.5 needs the two
towers to attend to each other inside every layer, and because the action
expert replaces plain RMSNorm with adaptive RMSNorm (AdaRMS) conditioned on
the flow-matching timestep.

The parameter tree deliberately matches the official checkpoints:
`paligemma.*` for the VLM and `gemma_expert.*` for the expert, with the
expert's norms exposing a single `dense` projection producing (scale, shift,
gate).
"""

import torch
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

# Gemma variants, see openpi `gemma.py: get_config`.
GEMMA_VARIANTS = {
    "gemma_300m": {
        "width": 1024,
        "depth": 18,
        "mlp_dim": 4096,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
    },
    "gemma_2b": {
        "width": 2048,
        "depth": 18,
        "mlp_dim": 16_384,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
    },
}

# Softmax mask fill value used by big_vision / openpi, see gemma/modules.py.
BIG_NEG = -2.3819763e38


def get_gemma_variant(variant: str) -> dict:
    if variant not in GEMMA_VARIANTS:
        raise ValueError(f"Unknown gemma variant: {variant}. Expected one of {list(GEMMA_VARIANTS)}.")
    return GEMMA_VARIANTS[variant]


def apply_rope(x, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def gated_residual(x, y, gate):
    """`x + y` when the norm produced no gate, `x + y * gate` when it did."""
    if gate is None:
        return x + y
    return x + y * gate


class AdaRMSNorm(nn.Module):
    """RMSNorm whose scale/shift/gate are predicted from a conditioning vector.

    pi0.5 feeds the flow-matching timestep embedding as `cond`, so every action
    expert layer is modulated by the denoising step (openpi's `AdaRMSNorm`).
    A single `dense` layer emits the three modulation vectors, which is what the
    official checkpoints store: `dense.weight` of shape [3 * width, cond_dim].

    `cond` is either [B, cond_dim] (one timestep for the whole chunk) or
    [B, T, cond_dim] (a timestep per chunk position, which training-time RTC
    needs so prefix positions can be driven at t=0 while the rest denoise).

    Returns `(output, gate)`; the gate is applied by the caller on the residual
    branch, see `gated_residual`.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.cond_dim = cond_dim
        if cond_dim is None:
            raise ValueError("`AdaRMSNorm` requires `cond_dim`; use the stock RMSNorm otherwise.")
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)

    def _norm(self, x):
        # Variance in float32 to match the reference implementation.
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")

        dtype = x.dtype
        normed = self._norm(x)

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if x.ndim == 3 and modulation.ndim == 2:
            # One timestep for the whole sequence: broadcast over positions.
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)

        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, cond_dim={self.cond_dim}"


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PI05PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        **kwargs,
    ):
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm = get_gemma_variant(paligemma_variant)
        expert = get_gemma_variant(action_expert_variant)

        self.paligemma_config = CONFIG_MAPPING["paligemma"](
            transformers_version="4.48.1",
            _vocab_size=257152,
            bos_token_id=2,
            eos_token_id=1,
            hidden_size=vlm["width"],
            image_token_index=257152,
            model_type="paligemma",
            pad_token_id=0,
            projection_dim=vlm["width"],
            text_config={
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": vlm["width"],
                "intermediate_size": vlm["mlp_dim"],
                "model_type": "gemma",
                "num_attention_heads": vlm["num_heads"],
                "num_hidden_layers": vlm["depth"],
                "num_image_tokens": 256,
                "num_key_value_heads": vlm["num_kv_heads"],
                "head_dim": vlm["head_dim"],
                "torch_dtype": "float32",
                "vocab_size": 257152,
            },
            vision_config={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "model_type": "siglip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "num_image_tokens": 256,
                "patch_size": 14,
                "projection_dim": vlm["width"],
                "projector_hidden_act": "gelu_fast",
                "torch_dtype": "float32",
                "vision_use_head": False,
            },
        )

        self.gemma_expert_config = CONFIG_MAPPING["gemma"](
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=2,
            eos_token_id=1,
            head_dim=expert["head_dim"],
            hidden_act="gelu_pytorch_tanh",
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=expert["width"],
            initializer_range=0.02,
            intermediate_size=expert["mlp_dim"],
            max_position_embeddings=8192,
            model_type="gemma",
            num_attention_heads=expert["num_heads"],
            num_hidden_layers=expert["depth"],
            num_key_value_heads=expert["num_kv_heads"],
            pad_token_id=0,
            rms_norm_eps=1e-06,
            rope_theta=10000.0,
            torch_dtype="float32",
            transformers_version="4.48.1",
            use_cache=True,
            vocab_size=257152,
        )

        super().__init__(**kwargs)


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig, precision: str = "float32"):
        super().__init__(config=config)
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # The expert consumes action embeddings, never token ids.
        self.gemma_expert.model.embed_tokens = None

        self._convert_expert_norms_to_adarms()
        self.set_precision(precision)
        self.set_requires_grad()

    def _convert_expert_norms_to_adarms(self):
        """Swap every expert RMSNorm for an AdaRMS conditioned on the timestep.

        Only the action expert is adaptive in pi0.5; the VLM tower keeps stock
        Gemma norms (its parameters are shared with the pretrained PaliGemma).
        """
        expert_model = self.gemma_expert.model
        width = self.config.gemma_expert_config.hidden_size
        eps = self.config.gemma_expert_config.rms_norm_eps

        for layer in expert_model.layers:
            layer.input_layernorm = AdaRMSNorm(width, eps=eps, cond_dim=width)
            layer.post_attention_layernorm = AdaRMSNorm(width, eps=eps, cond_dim=width)
        expert_model.norm = AdaRMSNorm(width, eps=eps, cond_dim=width)

    def set_precision(self, precision: str = "float32"):
        """Cast to `precision`, keeping the numerically fragile bits in float32.

        Following openpi's `to_bfloat16_for_selected_params`, the Gemma norms
        stay float32 in bfloat16 mode — including the action expert's AdaRMS
        `dense` layers, whose (scale, shift, gate) modulate every residual.

        The SigLIP tower is cast wholesale, unlike openpi which keeps its patch
        and position embeddings in float32: `transformers`' SigLIP applies
        LayerNorm directly to the embedding output, so a float32 embedding
        feeding bfloat16 norms raises a dtype error.
        """
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        self.to(dtype=torch.bfloat16)

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    @property
    def language_model(self):
        """The Gemma decoder stack of the VLM, across transformers layouts."""
        lm = self.paligemma.language_model
        return lm.model if hasattr(lm, "model") else lm

    def embed_image(self, image: torch.Tensor):
        if hasattr(self.paligemma, "get_image_features"):
            return self.paligemma.get_image_features(image)
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        """Run both towers layer by layer with a shared attention over the concatenation.

        `inputs_embeds` is `[prefix_embs, suffix_embs]`; either entry may be
        None (prefix-only when filling the KV cache, suffix-only when denoising
        against a cached prefix). `adarms_cond` follows the same two-slot layout.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Read by `eager_attention_forward` so attention is only collected during
        # denoising, not during the prefix prefill (see `server.py`'s SAVE_ATTN).
        self._fill_kv_cache = fill_kv_cache

        models = [self.language_model, self.gemma_expert.model]

        batch_size = None
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.config.paligemma_config.text_config.num_hidden_layers
        head_dim = self.config.paligemma_config.text_config.head_dim

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]

                if adarms_cond[i] is not None:
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                else:
                    hidden_states, gate = layer.input_layernorm(hidden_states), None
                gates.append(gate)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B, L, H, D with L the sequence length, H the number of heads, D the head dim.
            # Concatenate along the token axis so both towers attend to each other.
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            att_output = self.eager_attention_forward(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )

            outputs_embeds = []
            start = 0
            gate_idx = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue

                layer = models[i].layers[layer_idx]
                end = start + hidden_states.shape[1]

                out_emb = layer.self_attn.o_proj(
                    att_output[:, start:end].to(layer.self_attn.o_proj.weight.dtype)
                )

                # First residual, gated by the input norm when adaptive.
                out_emb = gated_residual(hidden_states, out_emb, gates[gate_idx])
                after_first_residual = out_emb.clone()

                if adarms_cond[i] is not None:
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                else:
                    out_emb, gate = layer.post_attention_layernorm(out_emb), None

                out_emb = layer.mlp(out_emb.to(layer.mlp.up_proj.weight.dtype))

                # Second residual.
                out_emb = gated_residual(after_first_residual, out_emb, gate)

                outputs_embeds.append(out_emb)
                start = end
                gate_idx += 1

            inputs_embeds = outputs_embeds

        # Final norm.
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            if adarms_cond[i] is not None:
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
            else:
                out_emb = models[i].norm(hidden_states)
            outputs_embeds.append(out_emb)

        return outputs_embeds, past_key_values

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        # Repeat the KV heads to match the number of query heads (GQA / MQA).
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Upcast to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
        key_states = key_states.to(dtype=torch.float32).transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, BIG_NEG)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)

        # Optional attention capture for `server.py`'s overlay (SAVE_ATTN=1).
        # Layers are averaged with linearly increasing weights so the later,
        # more semantic layers dominate. Same scheme as pi0's.
        if getattr(self, "_save_attn", False) and not getattr(self, "_fill_kv_cache", False):
            p = probs.detach().cpu()
            if getattr(self, "_attn_probs_sum", None) is None:
                self._attn_probs_sum = p.clone()
                self._attn_weight_sum = 1.0
                self._attn_layer_idx = 1
            else:
                self._attn_layer_idx += 1
                w = float(self._attn_layer_idx)
                self._attn_probs_sum += p * w
                self._attn_weight_sum += w
            self._attn_probs = self._attn_probs_sum / self._attn_weight_sum

        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # -1 because the sequence length varies between prefill and denoising.
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
