from typing import Dict
import einops
from termcolor import colored
import torch
import os
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import asdict
from pprint import pformat
from gx_infer import base_policy as _base_policy
from gx_infer.websocket_policy_server import WebsocketPolicyServer

from lerobot.common.datasets.factory import IMAGENET_STATS
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.hvla.configuration_hvla import HVLAConfig
from lerobot.common.policies.hvla.modeling_hvla import HVLAPolicy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

HOST = "localhost"
PORT = "8001"

def process_img(img):
    img = torch.from_numpy(img)
    if img.ndim == 3 or img.ndim == 4 and img.shape[0] != 1:
        img = img.unsqueeze(0)
    img = einops.rearrange(img, "... h w c -> ... c h w").contiguous()
    img = img.type(torch.float32)
    img /= 255
    return img
    
class ServerPolicyDual(_base_policy.BasePolicy):
    def __init__(self, model: HVLAPolicy, device: str):
        self.model = model
        self.device = device

    def infer(self, obs_dict: Dict) -> Dict: 
        fast_input = dict()
        if "task" in obs_dict:
            slow_input = dict()
            for key in obs_dict:
                if "images" in key:
                    slow_input[key] = process_img(np.array(obs_dict[key])).to(self.device)
            slow_input['task'] = [obs_dict['task']]
            slow_input["task_index"] = torch.tensor(0).unsqueeze(0).to(self.device)

        if obs_dict['reset']:
            self.model.reset()
            self.model.s2_infer(slow_input) # block for first result
        elif "task" in obs_dict:
            self.model.s2_infer_async(slow_input)

        fast_input['observation.state'] = torch.tensor(np.array(obs_dict['observation.state'])).unsqueeze(0).float().to(self.device)

        action = self.model.s1_infer(fast_input).cpu().numpy()
        print(action.round(2))
        return {"actions": action }

@parser.wrap()
def main_wrapper(cfg: TrainPipelineConfig):
    cli_overrides = parser.get_cli_overrides("policy")
    cfg.policy = PreTrainedConfig.from_pretrained("./lerobot/common/policies/hvla", cli_overrides=cli_overrides)
    logging.info(pformat(asdict(cfg)))
    # Check device is available
    cfg.policy.device = "cuda" if torch.cuda.is_available() else "cpu"
    # configuration
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info("load dataset metainfo")
    ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )

    # load policy
    logging.info("Making policy.")
    cfg.policy.eval = True
    cfg.policy.load_path = parser.get_path_arg("policy")

    network: HVLAPolicy = HVLAPolicy(
        config=cfg.policy,
        dataset_stats=ds_meta
    )
    network.eval()

    policy = ServerPolicyDual(model=network, device=cfg.policy.device)

    policy_server = WebsocketPolicyServer(policy=policy, host=HOST, port=PORT)
    print(f"Starting server on {HOST}:{PORT}")
    policy_server.serve_forever()
    


if __name__ == '__main__':
    init_logging()
    main_wrapper()