#!/usr/bin/env python

"""
Example of training pi0_multi policy with 4 historical frames + current frame.

This example demonstrates how to use the pi0_multi policy with multi-frame inputs.
The policy will use 4 historical frames and 1 current frame, totaling 5 frames.
"""

import torch
from torch.optim import AdamW

from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.pi0_multi.configuration_pi0_multi import PI0MultiConfig


def main():
    # Create pi0_multi configuration
    config = PI0MultiConfig(
        # Multi-frame settings
        n_obs_steps=5,  # 4 historical frames + 1 current frame
        empty_cameras=4,  # Reserve slots for historical frames
        
        # Model settings
        chunk_size=50,
        n_action_steps=50,
        proj_width=1024,
        
        # Training settings
        optimizer_lr=2.5e-5,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )
    
    # Initialize policy
    policy_class = get_policy_class("pi0_multi")
    policy = policy_class(config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.train()
    
    print(f"Initialized pi0_multi policy with {config.n_obs_steps} observation steps")
    print(f"Empty cameras reserved: {config.empty_cameras}")
    
    # Example training setup
    optimizer = AdamW(policy.get_optim_params(), lr=config.optimizer_lr)
    
    # In a real training scenario, you would:
    # 1. Load dataset with delta_timestamps
    # 2. Set up data loader
    # 3. Training loop
    
    # Example dataset configuration for multi-frame:
    delta_timestamps = {
        "observation.images.camera_0": [-0.4, -0.3, -0.2, -0.1, 0.0],  # 4 historical + current
        "observation.images.camera_1": [-0.4, -0.3, -0.2, -0.1, 0.0],
        "observation.state": [-0.4, -0.3, -0.2, -0.1, 0.0],
    }
    
    print(f"Example delta_timestamps: {delta_timestamps}")
    print("Policy is ready for multi-frame training!")


if __name__ == "__main__":
    main()