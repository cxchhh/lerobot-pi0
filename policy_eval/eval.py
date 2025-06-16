import argparse
from pathlib import Path
from tqdm import tqdm

from gx_sim.render.batch_task import RenderTask, BatchRendererConfig
from gx_sim.render.batch_isaacsim import IsaacSimConfig
from gx_sim.render.config import TableTopRenderTaskConfig

from gx_utils import log, fm
from gx_utils.dtype import Render, ListRender, Physics, ListPhysics, Scene, ListScene, ID, ListPlan, Box, Mesh
from gx_utils.magic.profiler import profiler
from gx_utils.constant import GROUND
from gx_utils.magic.video import VideoRecorder
task_cfg = TableTopRenderTaskConfig
render_cfg = BatchRendererConfig(
    num_envs=1,
    debug=0,
    isaac_sim=IsaacSimConfig(headless=1),
)

import os
import pathlib
import torch
import json

from sim_data_gen.policy_eval.G1VLA_env import G1VLAEnv
GX_STORAGE_PATH = os.getenv("GX_STORAGE_PATH")

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def main():
    render_task = RenderTask(task_cfg, render_cfg)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(args.checkpoint, 'rb'))    # TODO: load checkpoint here
    cfg = payload['cfg']    # TODO: load config here

    device = torch.device(args.device)
    policy = None   # TODO: load policy here
    policy.to(device)
    policy.eval()

    n_envs = 1  # TODO: due to system env problems, haven't implemented parallelization yet
    env = G1VLAEnv(n_envs=n_envs,
                render_task=render_task, 
                n_obs_steps=cfg.n_obs_steps,
                uid=args.uid,
                ii=args.ii,
                n_action_steps=cfg.n_action_steps,
                horizon=cfg.horizon,
                output_dir=args.output_dir.split('/')[-1]
            )
    env.reset()

    all_success = []
    num_evals = 1 # TODO: due to system env problems, haven't implemented parallelization yet
    for eval_idx in range(num_evals):
        done = False
        obs, _, _ = env.step(dict(action=np.zeros(19)[None,None,...]), step_physics=False)
        while not done:
            np_obs_dict = dict(obs) 
            # TODO: here is assumed the output of environment step is a dictionary, with all values being tensors.
            #       in which, the np_obs_dict['head_image'] is head image,
            #                     np_obs_dict['wrist_image'] is wrist image,
            #                     np_obs_dict['qpos'] is input preprioceptions corresponding to lerobot info's definition of observation.state. 
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # TODO: here is assumed the output of policy is a dictionary, with all values being tensors.
            #       in which, the action_dict['action'] value contains predicted action chunk tensor corresponding to lerobot info's definition of action
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            obs, success, done = env.step(np_action_dict)

        # save success or not
        all_success.append(env.check_success())

    log_data = dict()
    log_data['success'] = np.array(all_success).tolist()
    out_path = os.path.join(args.output_dir, f'eval_log_{args.uid}_{args.ii}.json')
    json.dump(log_data, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample layout")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--uid', default=None)
    parser.add_argument('--ii', default=0)
    parser.add_argument('--delta_closeness', default=False)
    args = parser.parse_args()

    main()