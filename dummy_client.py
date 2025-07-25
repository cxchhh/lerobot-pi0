import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro
from gx_infer.action_chunk_broker import ActionChunkBroker
from gx_infer.websocket_client_policy import WebsocketClientPolicy


class DummyAgent:
    def init(self, init_qpos):
        self.init_qpos = init_qpos

    def reset(self):
        timestamp = datetime.now().strftime("%Y-%m%d-%H:%M")
        print(f"[{timestamp}] Resetting agent")

    def apply_action(self, action_dict: dict):
        action = action_dict
        # Simulate applying action to the robot
        timestamp = datetime.now().strftime("%Y-%m%d-%H:%M")
        print(f"[{timestamp}] Applying action: {action}")

    def get_obs(self) -> dict:
        obs_state = np.random.randn(8)
        obs_img_head = np.random.randn(224, 400, 3) * 255
        obs_img_right_wrist = np.random.randn(224, 400, 3) * 255

        obs_dict = {
            "observation.state": obs_state,
            "observation.images.cam_head": obs_img_head,
            "observation.images.cam_right_wrist": obs_img_right_wrist,
            "task": "pick up NaiLong and lift it up.",
            "reset": 0
        }
        return obs_dict


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8001

    max_hz = 1
    action_horizon: int = 5

    num_episodes: int = 1
    max_episode_steps: int = 30


def main(args: Args) -> None:
    ws_client_policy = WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    agent = DummyAgent()
    policy = ActionChunkBroker(
        policy=ws_client_policy,
        action_horizon=args.action_horizon,
    )

    last_log_time = time.time()
    agent.reset()
    time_buf = deque(maxlen=100)
    while True:
        t_start = time.time()
        obs_dict = agent.get_obs()
        action_dict = policy.infer(obs_dict)
        print(action_dict)
        agent.apply_action(action_dict)

        time_buf.append(time.time() - t_start)
        if t_start - last_log_time >= 1.0:
            print(f"Frequency: {1 / np.mean(time_buf):.1f} Hz")
            last_log_time = t_start


if __name__ == "__main__":
    tyro.cli(main)
