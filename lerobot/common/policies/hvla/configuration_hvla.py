from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("hvla")
@dataclass
class HVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 5 # k-frame observation
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    features : dict[str, PolicyFeature] = field(default_factory=lambda:{
        "observation.state": {
            "dtype": "float32",
            "shape": [30]
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": [
                3,
                224,
                400
            ],
            "names": [
                "channels",
                "height",
                "width"
            ],
            "info": {
                "video.height": 224,
                "video.width": 400,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 50,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": [
                3,
                224,
                224
            ],
            "names": [
                "channels",
                "height",
                "width"
            ],
            "info": {
                "video.height": 224,
                "video.width": 224,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 50,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                23
            ],
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
        }
    })

    # Shorter state and action vectors will be padded
    max_state_dim: int = 128
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # # Attention utils
    # use_cache: bool = True
    # attention_implementation: str = "eager"  # or fa2, flex

    # ===== VLM config =====
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    rewrite_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    freeze_backbone: bool = True

    # =====  VLM query config  =====
    latent_dim: int = 256
    query_layers: int = 6
    query_num_heads: int = 4
    query_ff_dim_multiplier: int = 4
    query_dropout_rate: float = 0.1

    # ===== History Transformer config =====
    max_seq_len: int = 32
    embed_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    ff_dim_multiplier: int = 4
    dropout_rate: float = 0.1
    command_dim: int = 4

    # ===== Flow-Matching config =====
    t_embed_dim: int = 128
    flow_hidden_dim: int = 384
    flow_num_layers: int = 3
    cond_feat_dim: int = 256
    sample_steps: int = 16
    path: str = "rectified"

    obs_dim: int = 30 # 98
    act_dim: int = 59 # 23

    # ===== Training Hyperparameters =====
    total_timesteps: int = 20_000
    log_interval: int = 200
    reset_ratio: float = 1e-3
    termination_threshold: float = 0.5
    bf16: bool = False
    load_path: str = ""

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6


    # TODO: Add EMA

    def __post_init__(self):
        super().__post_init__()

        # TODO(Steven): Validate device and amp? in all policy configs?
        """Input validation (not exhaustive)."""
        # if self.n_action_steps > self.chunk_size:
        #     raise ValueError(
        #         f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
        #         f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
        #     )
        # if self.n_obs_steps != 1:
        #     raise ValueError(
        #         f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
        #     )
        self.pretrained_path = None

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # for i in range(self.empty_cameras):
        #     key = f"observation.images.empty_camera_{i}"
        #     empty_camera = PolicyFeature(
        #         type=FeatureType.VISUAL,
        #         shape=(3, 480, 640),
        #     )
        #     self.input_features[key] = empty_camera
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
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
        return list(range(-self.n_obs_steps+1, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(-self.n_obs_steps+1, 1)) # g.t. action for each parallel frame

    @property
    def reward_delta_indices(self) -> None:
        return None