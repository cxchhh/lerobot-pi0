#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import logging
from dataclasses import asdict
from pprint import pformat

import torch
from termcolor import colored

from lerobot.common.datasets.factory import IMAGENET_STATS
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

@parser.wrap()
def eval_main(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    
    logging.info("load dataset metainfo")
    ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
    if cfg.dataset.use_imagenet_stats:
        for key in ds_meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                ds_meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    cfg.dataset.image_transforms.enable = True
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) 
    )

    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta
    )
    policy.eval()

    image_keys = ds_meta.camera_keys

    # fake input 
    observation = {}
    for key in ds_meta.features.keys():
        dtype=eval(f'torch.{ds_meta.features[key]["dtype"] if ds_meta.features[key]["dtype"] != "video" else "float32"}')
        if "index" in key or key in image_keys:
            observation[key] = torch.zeros(ds_meta.features[key]['shape'], dtype=dtype).unsqueeze(0).to(cfg.policy.device)
        else:
            observation[key] = torch.rand(ds_meta.features[key]['shape'], dtype=dtype).unsqueeze(0).to(cfg.policy.device)
        
        if key in image_keys and image_transforms:
            observation[key] = image_transforms(observation[key])
        
    observation['task'] = ["example_task"]
    observation["task_index"] = torch.tensor(0).unsqueeze(0).to(cfg.policy.device)

    # model inference
    with torch.inference_mode():
        action = policy.select_action(observation)

    print(action.shape, action)
    breakpoint()


if __name__ == "__main__":
    init_logging()
    eval_main()
