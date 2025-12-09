import math
import time
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

from lerobot.common.policies.hvla.configuration_hvla import HVLAConfig


# ---------------- Rotary Embedding ----------------
class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding for attention (Su et al., 2021).
    Applies rotations to Q,K along the last dimension in head space.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_pos: int = 16384):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires head_dim to be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_pos = max_pos

        # Initialize cache
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _grow_if_needed(self, seq_len: int, device: torch.device):
        """Grow cache if needed, preserving existing values"""
        if seq_len <= self.cos_cached.size(0) and self.cos_cached.device == device:
            return

        # Calculate new cache size (at least seq_len, but grow by chunks)
        new_size = max(seq_len, self.cos_cached.size(0) * 2, self.max_pos)
        new_size = min(new_size, self.max_pos)

        t = torch.arange(new_size, device=device)
        freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq.to(device))
        cos_new = torch.cos(freqs)
        sin_new = torch.sin(freqs)

        # Preserve existing cache if it exists
        if self.cos_cached.size(0) > 0:
            cos_new[:self.cos_cached.size(0)] = self.cos_cached
            sin_new[:self.sin_cached.size(0)] = self.sin_cached

        self.cos_cached = cos_new
        self.sin_cached = sin_new

    def get_cos_sin(self, seq_len: int, device: torch.device, offset: int = 0):
        """Get cos and sin for the given sequence length and offset"""
        self._grow_if_needed(seq_len + offset, device)
        cos = self.cos_cached[offset:offset+seq_len]  # [L, D/2]
        sin = self.sin_cached[offset:offset+seq_len]  # [L, D/2]
        return cos, sin

def _rotate_half(x):
    # split last dim (D) into (D/2, D/2) and apply (x1, x2) -> (-x2, x1)
    x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q, k: [B, H, L, D] - query and key tensors
        cos, sin: [L, D/2] - cosine and sine values for each position

    Returns:
        q_rot, k_rot: [B, H, L, D] - rotated query and key tensors
    """
    # Efficiently expand cos and sin to full dimension
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]

    # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin

    return q_rot, k_rot


# ---------------- Transformer Modules ----------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        batch_size, sequence_length, embed_dim = x.size()

        # Project to Q, K, V
        Q = self.wq(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (batch_size, num_heads, sequence_length, head_dim) @ (batch_size, num_heads, head_dim, sequence_length)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (batch_size, num_heads, sequence_length, head_dim)
        output = torch.matmul(attn_weights, V)

        # Concatenate heads and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embed_dim)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class CrossAttention(nn.Module):
    """
    Multi‑head cross‑attention layer without causal (look‑ahead) masking.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout probability applied to attention weights
                              and the output projection.
    """

    def __init__(self, embed_dim: int, kv_embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # scaling factor for dot‑product attention

        # Linear projections to obtain Q, K, V
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(kv_embed_dim, embed_dim)
        self.wv = nn.Linear(kv_embed_dim, embed_dim)

        # Output projection
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            x: torch.Tensor,  # (batch_size, tgt_len, embed_dim) — queries
            context: torch.Tensor,  # (batch_size, src_len, embed_dim) — keys/values
            attn_mask: torch.Tensor = None  # optional padding mask (1 = keep, 0 = mask)
    ) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (batch_size, tgt_len, embed_dim)
        """
        bsz, tgt_len, _ = x.size()
        src_len = context.size(1)

        # Linear projection + reshape to (B, H, L, D_h)
        Q = self.wq(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(context).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(context).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention scores: (B, H, T, S)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Optional: apply external mask (e.g., padding)
        if attn_mask is not None:
            # attn_mask == 0 indicates positions to mask
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))

        # Attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (B, H, T, D_h)
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads and apply output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, -1)  # (B, T, embed_dim)
        )
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, rotary_emb: RotaryEmbedding = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires head_dim even"
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

        self.rotary_emb = rotary_emb  # may be shared across blocks

    def forward(self, x, pos_offset: int = 0):
        # x: [B, L, E]
        B, L, E = x.size()

        # Project & reshape
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]

        # ---- RoPE here ----
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb.get_cos_sin(L, device=x.device, offset=pos_offset)
            q, k = apply_rope(q, k, cos, sin)

        # scaled dot-product attention + causal mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B,H,L,D]
        out = out.transpose(1, 2).contiguous().view(B, L, E)
        out = self.wo(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # GELU is a common activation in Transformers
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, rotary_emb: RotaryEmbedding = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout_rate, rotary_emb=rotary_emb)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, pos_offset: int = 0):
        x = x + self.attn(self.norm1(x), pos_offset=pos_offset)
        x = x + self.ff(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.ff_dim = config.embed_dim * config.ff_dim_multiplier
        self.dropout_rate = config.dropout_rate

        self.input_projection = nn.Linear(input_dim, config.embed_dim)

        # Shared rotary embedding for all blocks
        self.rotary_emb = RotaryEmbedding(
            dim=self.embed_dim // self.num_heads,
            base=10000.0,
            max_pos=getattr(config, 'max_pos', 16384)
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate, rotary_emb=self.rotary_emb)
             for _ in range(config.num_layers)]
        )
        self.output_layer = nn.Linear(config.embed_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the causal transformer.

        Args:
            x: [B, L, input_dim] - input tensor
            pos_offset: starting position index (useful for streaming with kv-cache)

        Returns:
            out: [B, L, output_dim] - output tensor
        """
        x = self.input_projection(x)  # [B, L, E]

        # Pass through transformer blocks
        for blk in self.transformer_blocks:
            x = blk(x, pos_offset=0)

        out = self.output_layer(x)  # [B, L, output_dim]
        return out

    def infer(self, x, pos_offset: int = 0):
        """
        Forward pass through the causal transformer.

        Args:
            x: [B, L, input_dim] - input tensor
            pos_offset: starting position index (useful for streaming with kv-cache)

        Returns:
            out: [B, L, output_dim] - output tensor
        """
        x = self.input_projection(x)  # [B, L, E]

        # Pass through transformer blocks
        for blk in self.transformer_blocks:
            x = blk(x, pos_offset=pos_offset)

        out = self.output_layer(x)  # [B, L, output_dim]
        return out


class QueryTransformerBlock(nn.Module):
    def __init__(self, embed_dim, kv_embed_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, kv_embed_dim, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout_rate)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), y)
        x = x + self.ff(self.norm2(x))
        return x


class QueryTransformer(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.q_dim = config.q_dim
        self.kv_dim = config.kv_dim
        self.query_layers = config.query_layers
        self.num_heads = config.query_num_heads
        self.ff_dim = config.q_dim * config.query_ff_dim_multiplier
        self.dropout_rate = config.query_dropout_rate

        # Query
        self.query = nn.Parameter(torch.zeros(1, 1, config.q_dim))  # learnable query vector

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [QueryTransformerBlock(self.q_dim, self.kv_dim, self.num_heads, self.ff_dim, self.dropout_rate)
             for _ in range(self.query_layers)]
        )

        # Normalization
        self.norm = nn.LayerNorm(config.q_dim)

        # Output layer
        self.output_layer = nn.Linear(config.q_dim, output_dim)

    def forward(self, y):
        # x shape: (batch_size, sequence_length, input_dim)

        # 1. Expand query to match batch size
        x = self.query.expand(y.size(0), -1, -1)  # (batch_size, 1, q_dim)

        # 2. Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, y)

        # 3. Normalize the output
        x = self.norm(x[:, 0])

        # 4. Predict the output
        output = self.output_layer(x)  # (batch_size, output_dim)

        return output


# ---------------- MLP Module ----------------
class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        hidden_dim = config.mlp_hidden_dim
        num_layers = config.mlp_num_layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------- Flow-Matching Modules ----------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal embedding for continuous time t in [0, 1].
    t: (batch_size, 1)
    return: (batch_size, dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=t.dtype) * (-math.log(10000.0) / max(half, 1)))
    args = t * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualMLPBlock(nn.Module):
    """
    LayerNorm -> GELU -> Linear residual block.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CondEncoder(nn.Module):
    def __init__(self, c_dim: int, out_dim: int = 512):
        super().__init__()
        self.token_ln = nn.LayerNorm(c_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(c_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.token_ln(cond)
        x = self.token_mlp(x)
        return x


class VectorFieldNet(nn.Module):
    """
    v_theta(x_t, t_emb, cond_emb) -> velocity field with shape of x_t.
    """

    def __init__(self, input_dim_x: int, input_dim_t: int, input_dim_c: int,
                 hidden_dim: int, num_layers: int):
        super().__init__()
        assert num_layers >= 2, "flow_num_layers should be >= 2"

        in_dim = input_dim_x + input_dim_t + input_dim_c
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.res_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(num_layers - 2)]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim_x),
        )

    def forward(self, x_flat: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_flat, t_emb, cond_emb], dim=-1)
        z = self.input_proj(z)
        for blk in self.res_blocks:
            z = blk(z)
        return self.output_proj(z)

class TransformerVectorField(nn.Module):
    """
    v_theta(X_t, t_emb, cond_emb) -> velocity field over a length-H action sequence.
    X_t:        (B, H, D_action)
    t_emb:      (B, H, t_dim)
    cond_emb:   (B, cond_dim)
    return:     (B, H, D_action)
    """

    def __init__(
        self,
        action_dim: int,
        t_dim: int,
        cond_dim: int,
        hidden_dim: int,
        horizon: int,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.t_proj = nn.Linear(t_dim, hidden_dim)
        self.c_proj = nn.Linear(cond_dim, hidden_dim)
        self.pos_emb = nn.Embedding(horizon, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.register_buffer("pos_ids", torch.arange(horizon).unsqueeze(0), persistent=False)

    def forward(
        self,
        x_seq: torch.Tensor,     # (B, H, D_action)
        t_emb: torch.Tensor,     # (B, H, t_dim)
        cond_emb: torch.Tensor,  # (B, cond_dim)
    ) -> torch.Tensor:
        B, H, D = x_seq.shape

        h = self.action_proj(x_seq)  # (B, H, hidden)

        # 时间 embedding：逐 token
        t_h = self.t_proj(t_emb)     # (B, H, hidden)

        # 条件 embedding：全局，再 broadcast 到 H
        c_h = self.c_proj(cond_emb).unsqueeze(1)  # (B, 1, hidden)
        c_h = c_h.expand(-1, H, -1)              # (B, H, hidden)

        h = h + t_h + c_h
        # Positional encodings
        h = h + self.pos_emb(self.pos_ids[:, :H])  # (B, H, hidden)
        
        h = self.encoder(h)  # (B, H, hidden)

        v = self.output_proj(h)  # (B, H, D_action)
        return v



class FlowMatching(nn.Module):
    """
    Flow-Matching policy that maps (condition, time) -> vector field over action.
    - condition: (batch_size, cond_feat_dim)
    - ground truth action: (batch_size, output_dim)
    - model internal action vector (batch_size, output_dim)
    """

    def __init__(self, config:HVLAConfig, input_dim, output_dim):
        super().__init__()
        self.config = config
        self.device = config.device
        self.dtype = torch.bfloat16 if config.bf16 else torch.float32

        # Derived dimensions
        self.action_dim = int(output_dim)
        self.cond_feat_dim = int(config.cond_feat_dim)
        self.horizon = int(config.n_action_steps)

        # Condition encoder
        self.condition_encoder = CondEncoder(input_dim, self.cond_feat_dim).to(self.device, dtype=self.dtype)
        self.delay_emb = nn.Embedding(self.horizon, self.config.cond_feat_dim).to(self.device, dtype=self.dtype)

        # Vector field network
        self.vector_field = TransformerVectorField(
            action_dim=self.action_dim,
            t_dim=self.config.t_embed_dim,
            cond_dim=self.config.cond_feat_dim,
            hidden_dim=self.config.flow_hidden_dim,
            horizon=self.horizon,
        ).to(self.device, dtype=self.dtype)

        self.streaming_buffer = None

    # --------- Condition encoding ---------
    def encode_condition(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        """
        condition_tokens: (batch_size, cond_feat_dim)
        return: (batch_size, condition_proj_dim)
        """
        projected = self.condition_encoder(condition_tokens)  # (B, P)
        return projected

    # --------- Flow-Matching training loss ---------
    def flow_matching_loss(self, target_action: torch.Tensor, condition_token: torch.Tensor, delay_steps: torch.Tensor) -> Tuple[
        torch.Tensor, dict]:
        """
        target_action: (batch_size, horizon, output_dim)
        condition_token:    (batch_size, cond_feat_dim)
        delay_steps:        (batch_size)
        """
        batch_size = target_action.shape[0]
        device = target_action.device

        # Sample time and noise
        time_scalar = torch.rand(batch_size, self.horizon, 1, device=device, dtype=self.dtype)  # U(0,1)
        gaussian_noise = torch.randn_like(target_action, device=device, dtype=self.dtype)

        if self.config.path == "rectified":
            # Straight path: x_t = (1 - t) x0 + t * eps; v* = eps - x0
            xt = (1.0 - time_scalar) * target_action + time_scalar * gaussian_noise
            velocity_target = gaussian_noise - target_action
        elif self.config.path == "cosine":
            # Gaussian path: alpha(t)=cos(pi/2 t), beta(t)=sin(pi/2 t)
            alpha = torch.cos(0.5 * math.pi * time_scalar)
            beta = torch.sin(0.5 * math.pi * time_scalar)
            xt = alpha * target_action + beta * gaussian_noise

            dalpha = -0.5 * math.pi * torch.sin(0.5 * math.pi * time_scalar)
            dbeta = 0.5 * math.pi * torch.cos(0.5 * math.pi * time_scalar)
            velocity_target = dalpha * target_action + dbeta * gaussian_noise
        else:
            raise ValueError(f"Unknown path type: {self.config.path}")

        # Time/condition embeddings
        time_scalar_flat = time_scalar.view(batch_size * self.horizon, 1)           # (B*H, 1)
        time_emb_flat = timestep_embedding(time_scalar_flat, self.config.t_embed_dim)  # (B*H, t_dim)
        time_emb = time_emb_flat.view(batch_size, self.horizon, -1).to(device=self.device, dtype=self.dtype) # (B, H, t_dim)
        delay_emb = self.delay_emb(delay_steps.to(device=self.device, dtype=torch.long))
        cond_emb = self.encode_condition(condition_token).to(device=self.device, dtype=self.dtype) + delay_emb # (B, condition_proj_dim)

        # Predict velocity and compute MSE
        velocity_pred = self.vector_field(xt, time_emb, cond_emb)
        loss = F.mse_loss(velocity_pred, velocity_target)

        with torch.no_grad():
            target_mse = F.mse_loss(velocity_target, torch.zeros_like(velocity_target, device=self.device, dtype=self.dtype))

        logs = {"loss": loss.item(), "target_var": target_mse.item()}
        return loss, logs

    # --------- Sampling (ODE integration: Euler) ---------
    @torch.no_grad()
    def sample(self, condition_token: torch.Tensor, delay_steps: torch.Tensor, steps: int = 16, deterministic=False) -> torch.Tensor:
        """
        condition_tokens: (batch_size, cond_feat_dim)
        return: (batch_size, n_action_steps, output_dim) in original (unnormalized) scale
        """
        batch_size = condition_token.shape[0]

        # Start from standard Normal at t=1
        if deterministic:
            action = torch.zeros(batch_size, self.horizon, self.action_dim, device=self.device, dtype=self.dtype)
        else:
            action = torch.randn(batch_size, self.horizon, self.action_dim, device=self.device, dtype=self.dtype)

        # Pre-compute condition embedding
        delay_emb = self.delay_emb(delay_steps.to(device=self.device, dtype=torch.long))
        cond_emb = self.encode_condition(condition_token) + delay_emb

        dt = 1.0 / steps
        # Integrate dx/dt = -v_theta(x, t, cond) from t=1 -> 0
        for step in range(steps, 0, -1):
            t_curr = torch.full((batch_size, 1), step * dt, device=self.device, dtype=self.dtype).clamp(max=1.0 - 1e-4)
            t_emb = timestep_embedding(t_curr, self.config.t_embed_dim).to(device=self.device, dtype=self.dtype)
            t_emb = t_emb.unsqueeze(1).expand(-1, self.horizon, -1)
            velocity = self.vector_field(action, t_emb, cond_emb)
            action = action - velocity * dt

        # De-normalize and reshape
        action = action[:, 0, :]
        return action
    
    @torch.no_grad()
    def streaming_reset(self, init_condition_token: torch.Tensor, deterministic=False):
        self.streaming_buffer = []

        # Start from standard Normal at t=1
        if deterministic:
            action = torch.zeros(1, 1, self.action_dim, device=self.device, dtype=self.dtype)
        else:
            action = torch.randn(1, 1, self.action_dim, device=self.device, dtype=self.dtype)

        # Pre-compute condition embedding
        delay_emb = self.delay_emb(torch.zeros(1, device=self.device, dtype=torch.long))
        cond_emb = self.encode_condition(init_condition_token) + delay_emb

        dt = 1.0 / self.horizon
        t_curr = (torch.arange(1, self.horizon + 1) * dt).clamp(max=1.0 - 1e-4).unsqueeze(-1) # [dt, 2dt, ..., 1]
        for i in range(1, self.horizon):
            self.streaming_buffer.append(action.clone())
            noisy_actions = torch.cat(self.streaming_buffer, dim=1)
            t_emb = timestep_embedding(t_curr[-i:], self.config.t_embed_dim).unsqueeze(0).to(device=self.device, dtype=self.dtype)
            velocity = self.vector_field(noisy_actions, t_emb, cond_emb)
            noisy_actions = noisy_actions - velocity * dt
            self.streaming_buffer = list(noisy_actions.chunk(noisy_actions.size(1), dim=1))

        self.streaming_buffer.append(action.clone())

    @torch.no_grad()
    def streaming_sample(self, condition_token: torch.Tensor, delay_steps: torch.Tensor, deterministic=False) -> torch.Tensor:
        """
        condition_tokens: (batch_size, cond_feat_dim)
        return: (batch_size, n_action_steps, output_dim) in original (unnormalized) scale
        """
        
        if len(self.streaming_buffer) == 0:
            self.streaming_reset(condition_token, deterministic)

        # Start from standard Normal at t=1
        if deterministic:
            action = torch.zeros(1, 1, self.action_dim, device=self.device, dtype=self.dtype)
        else:
            action = torch.randn(1, 1, self.action_dim, device=self.device, dtype=self.dtype)

        # Pre-compute condition embedding
        delay_emb = self.delay_emb(delay_steps.to(device=self.device, dtype=torch.long))
        cond_emb = self.encode_condition(condition_token) + delay_emb
        
        dt = 1.0 / self.horizon
        # Integrate dx/dt = -v_theta(x, t, cond) from t=1 -> 0
        t_curr = (torch.arange(1, self.horizon + 1) * dt).clamp(max=1.0 - 1e-4).unsqueeze(-1) # [dt, 2dt, ..., 1]
        t_emb = timestep_embedding(t_curr, self.config.t_embed_dim).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        noisy_actions = torch.cat(self.streaming_buffer, dim=1)
        velocity = self.vector_field(noisy_actions, t_emb, cond_emb)
        noisy_actions = noisy_actions - velocity * dt
        self.streaming_buffer = list(noisy_actions.chunk(noisy_actions.size(1), dim=1))

        out_action = self.streaming_buffer.pop(0)
        self.streaming_buffer.append(action.clone())
        return out_action.squeeze(1)

# ---------------- Transformer + Flow-Matching Modules ----------------
class TransformerFlowMatching(nn.Module):
    """
    Transformer + Flow-Matching policy that maps (observation history, time) -> vector field over action.
    - observation history: (batch_size, history_seq_len, input_dim)
    - ground truth action: (batch_size, act_dim)
    - model internal action vector (batch_size, act_dim)
    """

    def __init__(self, config:HVLAConfig, input_dim, output_dim):
        super().__init__()
        self.config = config

        # Derived dimensions
        self.action_dim = int(output_dim)

        # Causal temporal Transformer backbone
        self.backbone = CausalTransformer(config, input_dim=input_dim, output_dim=config.embed_dim)

        # Vector field network
        self.vector_field = VectorFieldNet(
            input_dim_x=self.action_dim,
            input_dim_t=self.config.t_embed_dim,
            input_dim_c=self.config.embed_dim,
            hidden_dim=self.config.flow_hidden_dim,
            num_layers=self.config.flow_num_layers,
        )

        # Action normalization buffers (z-score). Not persisted in checkpoints by default.
        self.register_buffer("action_mean", torch.zeros(1, self.action_dim), persistent=False)
        self.register_buffer("action_std", torch.ones(1, self.action_dim), persistent=False)

    # --------- Action normalization API ---------
    @torch.no_grad()
    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Set dataset-wide action normalization stats.
        mean, std: shape (action_dim,) or (1, action_dim)
        """
        mean = mean.view(1, -1).to(self.action_mean.device)
        std = std.view(1, -1).to(self.action_std.device).clamp(min=1e-6)
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    # --------- Flow-Matching training loss ---------
    def flow_matching_loss(self, target_action: torch.Tensor, observations: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        target_action: (batch_size, history_seq_len, output_dim)
        observations:    (batch_size, history_seq_len, input_dim)
        """
        B, S, _ = observations.shape
        device = target_action.device

        # Backbone features
        condition_tokens = self.backbone(observations)
        condition_tokens = condition_tokens.reshape(B * S, -1)

        # Normalize
        target_flat = target_action.reshape(B * S, -1)
        target_norm = (target_flat - self.action_mean) / self.action_std  # (B*S, D)

        # Sample time and noise
        time_scalar = torch.rand(B * S, 1, device=device)  # U(0,1)
        gaussian_noise = torch.randn_like(target_norm)

        if self.config.path == "rectified":
            # Straight path: x_t = (1 - t) x0 + t * eps; v* = eps - x0
            xt = (1.0 - time_scalar) * target_norm + time_scalar * gaussian_noise
            velocity_target = gaussian_noise - target_norm
        elif self.config.path == "cosine":
            # Gaussian path: alpha(t)=cos(pi/2 t), beta(t)=sin(pi/2 t)
            alpha = torch.cos(0.5 * math.pi * time_scalar)
            beta = torch.sin(0.5 * math.pi * time_scalar)
            xt = alpha * target_norm + beta * gaussian_noise

            dalpha = -0.5 * math.pi * torch.sin(0.5 * math.pi * time_scalar)
            dbeta = 0.5 * math.pi * torch.cos(0.5 * math.pi * time_scalar)
            velocity_target = dalpha * target_norm + dbeta * gaussian_noise
        else:
            raise ValueError(f"Unknown path type: {self.config.path}")

        # Time/condition embeddings
        time_emb = timestep_embedding(time_scalar, self.config.t_embed_dim)  # (B*S, t_embed_dim)

        # Predict velocity and compute MSE
        velocity_pred = self.vector_field(xt, time_emb, condition_tokens)
        loss = F.mse_loss(velocity_pred, velocity_target)

        with torch.no_grad():
            target_mse = F.mse_loss(velocity_target, torch.zeros_like(velocity_target))

        logs = {"loss": loss.item(), "target_var": target_mse.item()}
        return loss, logs

    # --------- Sampling (ODE integration: Euler) ---------
    @torch.no_grad()
    def sample(self, observations: torch.Tensor, steps: int = 16, deterministic: bool = False) -> torch.Tensor:
        """
        observations: (B, S, input_dim)
        return: (B, output_dim)
        """
        device = next(self.parameters()).device
        B, S, _ = observations.shape

        tokens = self.backbone(observations)  # (B, S, embed_dim)
        cond_emb = tokens.reshape(B * S, self.config.embed_dim)  # (B*S, C_cond)

        if deterministic:
            action = torch.zeros(B * S, self.action_dim, device=device)
        else:
            action = torch.randn(B * S, self.action_dim, device=device)

        dt = 1.0 / steps
        for step in range(steps, 0, -1):
            t_curr = torch.full((B * S, 1), step * dt, device=device).clamp(max=1.0 - 1e-4)  # (B*S, 1)
            t_emb = timestep_embedding(t_curr, self.config.t_embed_dim)  # (B*S, t_dim)
            velocity = self.vector_field(action, t_emb, cond_emb)  # (B*S, ·)
            action = action - velocity * dt

        action = action * self.action_std + self.action_mean
        action = action.reshape(B, S, self.action_dim)[:, -1, :]  # return the last action prediction
        return action
