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

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    """Configuration for pi0.5 (openpi `PI0Pytorch` with `pi05=True`).

    Field names mirror the official LeRobot pi05 config so that the `config.json`
    shipped with `lerobot/pi05_base` & friends parses as-is.

    Differences vs. `PI0Config` that matter here:
      - State enters through the *prompt* as 256 discrete bins, not through a
        `state_proj` into the suffix. Hence `max_state_dim` only controls padding
        before discretization, and `tokenizer_max_length` must be large enough to
        hold the state string (200 vs. pi0's 48).
      - Normalization is quantile-based (q01/q99), not mean/std.
      - The flow-matching timestep modulates the action expert through adaptive
        RMSNorm instead of being concatenated with the action embedding.
    """

    # Gemma variants: "gemma_2b" for the VLM, "gemma_300m" for the action expert.
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # "bfloat16" | "float32"

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50  # openpi calls this "action_horizon"
    n_action_steps: int = 50

    # Stride between predicted action frames, and how many past state frames to
    # feed. Both mirror `PI0Config` so the two policies can be given an identical
    # input/output contract: with n_plan_steps=3 the chunk covers
    # `chunk_size * 3` dataset frames instead of `chunk_size`, and n_obs_states=4
    # feeds states at t-9, t-6, t-3, t.
    n_plan_steps: int = 1
    n_obs_states: int = 1

    # Shorter state and action vectors will be padded to these dimensions.
    max_state_dim: int = 32
    max_action_dim: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Flow matching, see openpi `PI0Pytorch`.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Image preprocessing, see openpi `preprocessing_pytorch.py`.
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    # Add empty (black, masked-out) images for cameras absent from the batch.
    empty_cameras: int = 0

    # Tokenizer. Must fit the whole prompt: roughly 4 tokens per state dimension,
    # times `n_obs_states` frames, plus the task text. 200 covers one 32-dim
    # frame; 4x40 dims needs ~600.
    tokenizer_max_length: int = 200

    # Training-time RTC (arXiv:2512.05964), same semantics as `PI0Config`.
    # Sample a prefix length per batch element; those positions are fed clean
    # (no noise, no loss) so the model learns to condition on already-committed
    # actions. At inference, set x_t[:d] = action_prefix and time[:d] = 0 — no
    # ΠGDM inpainting overhead.
    train_time_rtc: bool = False
    train_time_rtc_max_prefix_frac: float = 0.5
    """Upper bound on prefix length as a fraction of chunk (paper's d <= H - s
    constraint; 0.5 means d in [0, chunk_size//2])."""
    train_time_rtc_prefix_drop_p: float = 0.0
    """Per-batch-element probability of FORCING d=0 (no prefix), on top of the
    uniform sample. Classifier-free-guidance-style dropout so the model can
    still plan from scratch when no prefix is available."""

    # Attention utils. pi0.5 ships a self-contained eager attention (matching
    # openpi's upcast-to-float32 softmax); no other backend is wired up.
    use_cache: bool = True

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Optimizer settings, see openpi `AdamW`.
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Accepted-and-ignored: these live in the `config.json` of the official
    # checkpoints (they belong to the upstream `PreTrainedConfig`/`HubMixin`,
    # which this fork predates). Declared so draccus can parse those files.
    push_to_hub: bool = True
    repo_id: str | None = None
    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")
        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        if self.n_plan_steps > 0:
            return list(range(0, self.chunk_size * self.n_plan_steps, self.n_plan_steps))
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def state_delta_indices(self) -> list:
        return list(range(-(self.n_obs_states - 1) * self.n_plan_steps, 1, self.n_plan_steps))
