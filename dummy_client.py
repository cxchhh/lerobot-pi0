import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro
from gx_infer.action_chunk_broker import AsyncActionChunkBroker, ActionChunkBroker
from gx_infer.websocket_client_policy import WebsocketClientPolicy


class DummyAgent:
    def init(self, init_qpos):
        self.init_qpos = init_qpos

    def reset(self):
        timestamp = datetime.now().strftime("%Y-%m%d-%H:%M")
        print(f"[{timestamp}] Resetting agent")

    def apply_action(self, action_dict: dict):
        action = action_dict['actions']
        print(f"action shape: {action.shape}")
        # Simulate applying action to the robot
        timestamp = datetime.now().strftime("%Y-%m%d-%H:%M")
        # print(f"[{timestamp}] Applying action: {action}")

    def get_obs(self, reset: int = 0) -> dict:
        obs_state = np.random.randn(65)
        obs_img_head = (np.random.rand(224,400,3) * 255).astype(np.uint8)
        obs_img_right_wrist = (np.random.rand(224,224,3) * 255).astype(np.uint8)

        obs_dict = {
            "observation.state": obs_state,
            "observation.images.head": obs_img_head,
            "observation.images.right_wrist": obs_img_right_wrist,
            "task": "Pick up the yellow block and place it on the table",
            "reset": reset
        }
        return obs_dict
    
    def get_obs_fast(self) -> dict:
        obs_state = np.random.randn(65)
        obs_dict = {
            "observation.state": obs_state,
            "reset": 0
        }
        return obs_dict


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8001

    action_horizon: int = 20


def main(args: Args) -> None:
    ws_client_policy = WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    agent = DummyAgent()
    policy = ws_client_policy

    last_log_time = time.time()
    agent.reset()
    time_buf = deque(maxlen=100)
    policy.infer(agent.get_obs(reset=1))
    _step = 0
    while True:
        t_start = time.time()
        if _step % args.action_horizon == 0:
            obs_dict = agent.get_obs()
        else: 
            obs_dict = agent.get_obs_fast()
        
        action_dict = policy.infer(obs_dict)
        agent.apply_action(action_dict)
        

        time_buf.append(time.time() - t_start)
        if t_start - last_log_time >= 0.5:
            print(f"Frequency: {(1 / np.mean(time_buf)):.2f} Hz")
            last_log_time = t_start

        time.sleep(max(0.02 - (time.time() - t_start), 0))
        _step += 1


if __name__ == "__main__":
    tyro.cli(main)
