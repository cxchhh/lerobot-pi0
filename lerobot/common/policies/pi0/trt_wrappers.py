"""ONNX-exportable wrappers of the PI0 inference path.

Two nn.Modules expose static-shape, dict-free, mutation-free forward paths so
`torch.onnx.export(dynamo=True)` can trace them cleanly:

* ``PrefillWrapper``   — runs SigLIP + PaliGemma language over the prefix and
                          returns the prefill KV cache as a single stacked tensor.
* ``DiffusionStepWrapper`` — one flow-matching step of the Gemma expert given
                              the prefill KV cache.

These reuse the trained weights of ``PaliGemmaWithExpertModel`` but re-implement
the transformer stack to eliminate features that break dynamo tracing / TRT
compile:

* dict mutation on ``past_key_values`` — replaced by a single stacked tensor.
* ``if hidden_states is None`` branching — the prefill and denoise paths are
  written as two dedicated forwards; each is straight-line.
* ``self._fill_kv_cache`` attribute mutation — no longer needed.
* fp64 sinusoidal positional embeddings — replaced by an fp32 version (error
  vs fp64 is well under the bf16 noise floor).
* in-place slice assignment in ``apply_rope`` — replaced by ``torch.cat``.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Static shapes: batch=1, two cameras (head + right_wrist), 48 lang tokens,
# 50 action steps. Kept as module-level for documentation; runtime shapes are
# derived from the actual tensors, not from these constants.
PREFIX_LEN = 560  # 2 * 256 image tokens + 48 language tokens
NUM_LAYERS = 18
KV_HEADS = 1
HEAD_DIM = 256
NUM_ATT_HEADS = 8
NUM_KV_GROUPS = NUM_ATT_HEADS // KV_HEADS


def _apply_rope_functional(x: Tensor, positions: Tensor, max_wavelength: float = 10_000.0) -> Tensor:
    """Functional replacement for ``paligemma_with_expert.apply_rope``.

    Same math (fp32 sin/cos, then cast back), but the two output halves are
    concatenated via ``torch.cat`` instead of being written in place. The
    in-place pattern was rejected by ``torch.onnx.export(dynamo=True)`` on
    torch 2.7.
    """
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    device = x.device
    x_f32 = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x_f32.split(d_half, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype)


def _sinusoidal_pos_embedding_fp32(
    time: Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """FP32 replacement for ``create_sinusoidal_pos_embedding``.

    The original uses fp64 through ``get_safe_dtype`` on CUDA. TRT rejects fp64
    constants. The delta from fp64 for these frequencies is far below the
    downstream bf16 noise floor.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _make_att_2d_masks_int(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Bit-equivalent to ``modeling_pi0.make_att_2d_masks`` but with an
    explicit int32 cumsum (dynamo prefers this over letting bool cumsum
    silently promote to int64)."""
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d & pad_2d


def _prefill_layer_stack(
    paligemma_with_expert,
    hidden: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
) -> Tensor:
    """Run the 18-layer PaliGemma language stack on prefix embs, returning a
    single stacked past_kv tensor of shape (L, 2, B, T, KV, D) in bf16.

    Straight-line replacement for ``PaliGemmaWithExpertModel.forward`` with
    ``inputs_embeds=[embs, None], fill_kv_cache=True``.
    """
    lm = (
        paligemma_with_expert.paligemma.language_model
        if not hasattr(paligemma_with_expert.paligemma.language_model, "model")
        else paligemma_with_expert.paligemma.language_model.model
    )
    text_cfg = paligemma_with_expert.paligemma.config.text_config
    num_layers = text_cfg.num_hidden_layers
    head_dim = text_cfg.head_dim
    num_att_heads = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    num_kv_groups = num_att_heads // num_kv_heads

    batch_size = hidden.shape[0]
    big_neg = -2.3819763e38

    ks_list, vs_list = [], []
    for layer_idx in range(num_layers):
        layer = lm.layers[layer_idx]
        h = layer.input_layernorm(hidden)
        input_shape = h.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        h_bf = h.to(dtype=torch.bfloat16)
        q = layer.self_attn.q_proj(h_bf).view(hidden_shape)
        k = layer.self_attn.k_proj(h_bf).view(hidden_shape)
        v = layer.self_attn.v_proj(h_bf).view(hidden_shape)
        q = _apply_rope_functional(q, position_ids)
        k = _apply_rope_functional(k, position_ids)
        ks_list.append(k)
        vs_list.append(v)

        # attention: expand K/V to per-head, then fp32 softmax
        seq_len = k.shape[1]
        k_g = k[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        k_g = k_g.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)
        v_g = v[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        v_g = v_g.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)
        q32 = q.to(torch.float32).transpose(1, 2)
        k32 = k_g.to(torch.float32).transpose(1, 2)
        att = torch.matmul(q32, k32.transpose(2, 3)) * (head_dim ** -0.5)
        att = torch.where(attention_mask[:, None, :, :], att, torch.tensor(big_neg, dtype=att.dtype, device=att.device))
        probs = torch.nn.functional.softmax(att, dim=-1).to(dtype=v_g.dtype)
        att_out = torch.matmul(probs, v_g.permute(0, 2, 1, 3))
        att_out = att_out.permute(0, 2, 1, 3).reshape(
            batch_size, -1, num_kv_heads * num_kv_groups * head_dim
        ).to(dtype=torch.bfloat16)

        if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
            att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
        proj = layer.self_attn.o_proj(att_out)
        # Native forward uses in-place `out_emb += hidden`, which does the add
        # in the higher-precision dtype (fp32 hidden) and then stores back as
        # bf16 (bf16 out_emb). Simulate this out-of-place: add first (promotes
        # to fp32 on the first layer where hidden is still fp32) then cast
        # back to proj dtype. From layer 1 onward hidden is already bf16 and
        # this is a no-op cast.
        out = (proj + hidden).to(proj.dtype)
        residual = out
        out = layer.post_attention_layernorm(out)
        out = layer.mlp(out)
        hidden = out + residual

    ks = torch.stack(ks_list, dim=0)  # (L, B, T, KV, D)
    vs = torch.stack(vs_list, dim=0)
    past_kv = torch.stack([ks, vs], dim=1)  # (L, 2, B, T, KV, D)
    return past_kv


def _denoise_layer_stack(
    paligemma_with_expert,
    hidden: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    past_kv: Tensor,
) -> Tensor:
    """Run the 18-layer Gemma expert stack on suffix embs with cached K/V.

    Straight-line replacement for ``PaliGemmaWithExpertModel.forward`` with
    ``inputs_embeds=[None, embs], past_key_values=<stacked>``.

    ``past_kv`` shape: (L, 2, B, prefix_seq, KV, D) bf16.
    """
    ge = paligemma_with_expert.gemma_expert.model
    text_cfg = paligemma_with_expert.paligemma.config.text_config
    num_layers = text_cfg.num_hidden_layers
    head_dim = text_cfg.head_dim
    num_att_heads = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    num_kv_groups = num_att_heads // num_kv_heads

    batch_size = hidden.shape[0]
    big_neg = -2.3819763e38

    for layer_idx in range(num_layers):
        layer = ge.layers[layer_idx]
        h = layer.input_layernorm(hidden)
        input_shape = h.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        h_bf = h.to(dtype=torch.bfloat16)
        q = layer.self_attn.q_proj(h_bf).view(hidden_shape)
        k = layer.self_attn.k_proj(h_bf).view(hidden_shape)
        v = layer.self_attn.v_proj(h_bf).view(hidden_shape)
        q = _apply_rope_functional(q, position_ids)
        k = _apply_rope_functional(k, position_ids)

        past_k = past_kv[layer_idx, 0]  # (B, prefix_seq, KV, D)
        past_v = past_kv[layer_idx, 1]
        k_all = torch.cat([past_k, k], dim=1)
        v_all = torch.cat([past_v, v], dim=1)

        seq_len = k_all.shape[1]
        k_g = k_all[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        k_g = k_g.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)
        v_g = v_all[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        v_g = v_g.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)
        q32 = q.to(torch.float32).transpose(1, 2)
        k32 = k_g.to(torch.float32).transpose(1, 2)
        att = torch.matmul(q32, k32.transpose(2, 3)) * (head_dim ** -0.5)
        att = torch.where(attention_mask[:, None, :, :], att, torch.tensor(big_neg, dtype=att.dtype, device=att.device))
        probs = torch.nn.functional.softmax(att, dim=-1).to(dtype=v_g.dtype)
        att_out = torch.matmul(probs, v_g.permute(0, 2, 1, 3))
        att_out = att_out.permute(0, 2, 1, 3).reshape(
            batch_size, -1, num_kv_heads * num_kv_groups * head_dim
        ).to(dtype=torch.bfloat16)

        if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
            att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
        proj = layer.self_attn.o_proj(att_out)
        out = (proj + hidden).to(proj.dtype)  # see _prefill_layer_stack for dtype note
        residual = out
        out = layer.post_attention_layernorm(out)
        out = layer.mlp(out)
        hidden = out + residual

    return ge.norm(hidden)


class PrefillWrapper(nn.Module):
    """Exportable prefix path: embed images + language → PaliGemma prefill.

    Inputs (all batch=1):
        img_head:  (1, 3, 224, 224) bf16   — first camera, SigLIP-ready
        img_right: (1, 3, 224, 224) bf16
        img_head_mask:  (1,) bool          — True if head camera present
        img_right_mask: (1,) bool
        lang_tokens: (1, 48) int64
        lang_masks:  (1, 48) bool

    Outputs:
        past_kv: (18, 2, 1, 560, 1, 256) bf16 — stacked K,V per layer
        prefix_pad_masks: (1, 560) bool
    """

    def __init__(self, flow_model):
        super().__init__()
        self.flow = flow_model
        self.paligemma_with_expert = flow_model.paligemma_with_expert

    def forward(
        self,
        img_head: Tensor,
        img_right: Tensor,
        img_head_mask: Tensor,
        img_right_mask: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ):
        embs_list = []
        pad_list = []
        att_flags: list[int] = []

        for img, msk in ((img_head, img_head_mask), (img_right, img_right_mask)):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)
            scale = torch.tensor(img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            img_emb = img_emb * scale
            bsize, num_img = img_emb.shape[:2]
            msk_bt = msk[:, None].expand(bsize, num_img)
            embs_list.append(img_emb)
            pad_list.append(msk_bt)
            att_flags += [0] * num_img

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs_list.append(lang_emb)
        pad_list.append(lang_masks)
        att_flags += [0] * lang_emb.shape[1]

        prefix_embs = torch.cat(embs_list, dim=1)
        prefix_pad = torch.cat(pad_list, dim=1)
        att_masks_t = torch.tensor(att_flags, dtype=torch.bool, device=prefix_embs.device)
        att_masks_e = att_masks_t[None, :].expand(prefix_embs.shape[0], -1)
        att_2d = _make_att_2d_masks_int(prefix_pad, att_masks_e)
        pos_ids = torch.cumsum(prefix_pad.to(torch.int32), dim=1) - 1

        past_kv = _prefill_layer_stack(self.paligemma_with_expert, prefix_embs, att_2d, pos_ids)
        return past_kv, prefix_pad


class DiffusionStepWrapper(nn.Module):
    """Exportable base single-step: one flow-matching velocity evaluation.

    Inputs:
        state:            (1, 35) fp32
        x_t:              (1, 50, 32) fp32
        timestep:         (1, 50) fp32
        past_kv:          (18, 2, 1, 560, 1, 256) bf16
        prefix_pad_masks: (1, 560) bool

    Outputs:
        v_t: (1, 50, 32) fp32
    """

    def __init__(self, flow_model):
        super().__init__()
        self.flow = flow_model
        self.paligemma_with_expert = flow_model.paligemma_with_expert
        self.state_proj = flow_model.state_proj
        self.action_in_proj = flow_model.action_in_proj
        self.action_out_proj = flow_model.action_out_proj
        self.action_time_mlp_in = flow_model.action_time_mlp_in
        self.action_time_mlp_out = flow_model.action_time_mlp_out
        self.n_action_steps = flow_model.config.n_action_steps
        self.proj_width = flow_model.config.proj_width

    def forward(
        self,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        past_kv: Tensor,
        prefix_pad_masks: Tensor,
    ) -> Tensor:
        state_emb = self.state_proj(state).to(dtype=torch.bfloat16)
        if state_emb.ndim != 3:
            state_emb = state_emb[:, None, :]
        bsize = state_emb.shape[0]
        state_len = state_emb.shape[1]

        # fp32 sinusoidal time emb (fp64 replacement)
        time_emb = _sinusoidal_pos_embedding_fp32(
            timestep.reshape(-1), self.proj_width, 4e-3, 4.0, device=state.device
        ).reshape(timestep.shape[0], timestep.shape[1], -1)
        time_emb = time_emb.to(dtype=state_emb.dtype)

        action_emb = self.action_in_proj(x_t)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        suffix_embs = torch.cat([state_emb, action_time_emb], dim=1)
        state_pad = torch.ones(bsize, state_len, dtype=torch.bool, device=state.device)
        action_pad = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=state.device)
        suffix_pad = torch.cat([state_pad, action_pad], dim=1)

        # suffix internal attention: state and action's first token attend
        # freely; remaining action tokens attend causally to the block.
        att_flags = [1] * state_len + [1] + [0] * (self.n_action_steps - 1)
        suffix_att = torch.tensor(att_flags, dtype=torch.bool, device=state.device)
        suffix_att = suffix_att[None, :].expand(bsize, -1)
        suffix_len = suffix_pad.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = _make_att_2d_masks_int(suffix_pad, suffix_att)
        full_att = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_off = torch.sum(prefix_pad_masks.to(torch.int32), dim=-1)[:, None]
        pos_ids = prefix_off + torch.cumsum(suffix_pad.to(torch.int32), dim=1) - 1

        hidden = _denoise_layer_stack(self.paligemma_with_expert, suffix_embs, full_att, pos_ids, past_kv)
        suffix_out = hidden[:, -self.n_action_steps:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t


class DiffusionStepBwdWrapper(nn.Module):
    """RTC F3 path: compute ∂v/∂x_t @ error via autograd.

    The wrapper runs the exact same forward as ``DiffusionStepWrapper`` but
    also returns the VJP against ``error``. It exists so a joint fwd+bwd
    graph can be captured by ``torch.export.export`` and then lowered to
    ONNX / TRT. Whether the export actually succeeds is validated at engine
    build time; on failure the caller falls back to PyTorch autograd.
    """

    def __init__(self, fwd_wrapper: DiffusionStepWrapper):
        super().__init__()
        self.fwd = fwd_wrapper

    def forward(
        self,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        past_kv: Tensor,
        prefix_pad_masks: Tensor,
        error: Tensor,
    ) -> Tensor:
        x_in = x_t.detach().requires_grad_(True)
        v = self.fwd(state, x_in, timestep, past_kv, prefix_pad_masks)
        (grad_x,) = torch.autograd.grad(v, x_in, grad_outputs=error, create_graph=False)
        return grad_x
