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

import numpy as np
import torch
from termcolor import colored

from gx_utils.transform import rot2euler, euler2rot

from sim_data_gen.G1.robot_cfg import G1_HOME_POSE, G1_JOINTS
import tqdm

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
from sim.scene import G1, fix_object_poses, init_scene, physics, scene, cams, SIM_PATH, robot, update_cams, update_dofs_position


def state_to_preprio(state):
    eef_T = robot.fk_link(np.concatenate([np.array(G1_HOME_POSE[:-19]), state]), link='R_hand_base_link')
    eef_trans = eef_T[0]
    try:
        eef_euler = rot2euler(eef_T[1])
    except Exception as e:
        print(eef_T[1])
    thumb_horizontal = np.array(state[7]).reshape(1)
    finger_closeness = np.array([
        ((state[8]) / 0.3 + state[9] / 0.4 + state[10] / 0.6) / 3,   # mean thumb closeness
        state[11:].mean() / 1.7  # mean 4-finger closeness
    ]).mean()
    finger_closeness = np.array(finger_closeness).reshape(1)
    step_preprioception = np.concatenate([eef_trans, eef_euler, finger_closeness], axis=-1)

    # check if hand is closed and has not grasped any object. If grasped object, the finger joint angle coupling coefficient would deviate from control target.
    reset_signal = False
    # if finger_closeness>0.6 and np.abs(state[11]/state[12]-1)<0.15 and np.abs(state[13]/state[14]-1)<0.15 and np.abs(state[15]/state[16]-1)<0.15 and np.abs(state[17]/state[18]-1)<0.15:
    #     reset_signal = True
    return step_preprioception, reset_signal

def make_observation(cfg, ds_meta, image_transforms, image_head, image_right_wrist, state):
    image_keys = ds_meta.camera_keys
    preprio, _ = state_to_preprio(state[-19:])
    observation = {}
    observation['observation.state'] = torch.concat([torch.tensor(state[:7]), torch.tensor(preprio)]).float().unsqueeze(0).to(cfg.policy.device)
    observation['observation.images.cam_head'] = torch.tensor(image_head).unsqueeze(0).to(cfg.policy.device)
    observation['observation.images.cam_right_wrist'] = torch.tensor(image_right_wrist).unsqueeze(0).to(cfg.policy.device)

    for key in ds_meta.features.keys():
        dtype=eval(f'torch.{ds_meta.features[key]["dtype"] if ds_meta.features[key]["dtype"] != "video" else "float32"}')
        if "index" in key or key in image_keys:
            observation[key] = torch.zeros(ds_meta.features[key]['shape'], dtype=dtype).unsqueeze(0).to(cfg.policy.device)
        if key in image_keys and image_transforms:
            observation[key] = image_transforms(observation[key])
        
    observation['task'] = ["example_task"]
    observation["task_index"] = torch.tensor(0).unsqueeze(0).to(cfg.policy.device)

    return observation




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

    init_scene()

    # fix other parts of the body
    # G1.set_dofs_kp(np.ones([34])*400,
    #             [G1.get_joint(name).dofs_idx_local[0] for name in G1_JOINTS[:34]])

    for cam in cams:
        cam.start_recording()
    with torch.inference_mode():
        for i in tqdm.tqdm(range(30)):
            state = torch.tensor([G1.get_dofs_position(G1.get_joint(name).dofs_idx_local) for name in G1_JOINTS]).cpu().numpy()
            update_cams(cams, state)
            image_head, *other = cams[0].render()
            image_right_wrist, *other = cams[1].render()

            # assemble inputs
            observation = make_observation(cfg, ds_meta, image_transforms, image_head, image_right_wrist, state)

            # policy inference
            action = policy.select_action(observation)

            # calc new positions
            (
                new_dofs_position,
                set_dofs_idx
            ) = update_dofs_position(state, action.squeeze(0).cpu().numpy())
            
            # step
            fix_object_poses(physics.obj_pose[0], False)

            G1.control_dofs_position(new_dofs_position, set_dofs_idx)
            G1.control_dofs_position(physics.traj_state[0][:34],
                [G1.get_joint(name).dofs_idx_local[0] for name in G1_JOINTS[:34]])
            print("now   :", state[34:41])
            print("target:", new_dofs_position[:7])
            print('force :', G1.get_dofs_control_force(set_dofs_idx)[:7].cpu().numpy())
            print('v    :', G1.get_dofs_velocity(set_dofs_idx)[:7].cpu().numpy())
            scene.step()
            
            
        
    for ci, cam in enumerate(cams):
        cam.stop_recording(save_to_filename=f'{SIM_PATH}/app/genesis/video_view{ci}.mp4', fps=60)


if __name__ == "__main__":
    init_logging()
    eval_main()
