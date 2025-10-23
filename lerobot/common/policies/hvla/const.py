import numpy as np

locomotion_list = [
    {"language": "move forward", "command": [0.35, 0, 0]},
    {"language": "move backward", "command": [-0.35, 0, 0]},
    {"language": "turn left", "command": [0, 0, 0.35]},
    {"language": "turn right", "command": [0, 0, -0.35]},
    {"language": "move to the left", "command": [0, 0.35, 0]},
    {"language": "move to the right", "command": [0, -0.35, 0]},
    {"language": "move forward fast", "command": [0.5, 0, 0]},
    {"language": "move backward fast", "command": [-0.5, 0, 0]},
    {"language": "turn left fast", "command": [0, 0, 0.5]},
    {"language": "turn right fast", "command": [0, 0, -0.5]},
    {"language": "move to the left fast", "command": [0, 0.5, 0]},
    {"language": "move to the right fast", "command": [0, -0.5, 0]},
    {"language": "move forward slowly", "command": [0.2, 0, 0]},
    {"language": "move backward slowly", "command": [-0.2, 0, 0]},
    {"language": "turn left slowly", "command": [0, 0, 0.2]},
    {"language": "turn right slowly", "command": [0, 0, -0.2]},
    {"language": "move to the left slowly", "command": [0, 0.2, 0]},
    {"language": "move to the right slowly", "command": [0, -0.2, 0]},
    {"language": "stop", "command": [0, 0, 0]},
    {"language": "stay static", "command": [0, 0, 0]},
]

body_list = [
    {"language": "stand up", "command": [1.05]},
    {"language": "crouch down fully", "command": [0.45]},
    {"language": "crouch down", "command": [0.45]},
    {"language": "crouch down half", "command": [0.75]},
    {"language": "Pick up things on the ground", "command": [0.45]},
    {"language": "Pick up things on the sofa", "command": [0.75]},
]

all_cmds = {
    "move forward": [0.35, 0, 0, 1.05],
    "move backward": [-0.35, 0, 0, 1.05],
    "turn left": [0, 0, 0.35, 1.05],
    "turn right": [0, 0, -0.35, 1.05],
    "move to the left": [0, 0.35, 0, 1.05],
    "move to the right": [0, -0.35, 0, 1.05],
    "move forward fast": [0.5, 0, 0, 1.05],
    "move backward fast": [-0.5, 0, 0, 1.05],
    "turn left fast": [0, 0, 0.5, 1.05],
    "turn right fast": [0, 0, -0.5, 1.05],
    "move to the left fast": [0, 0.5, 0, 1.05],
    "move to the right fast": [0, -0.5, 0, 1.05],
    "move forward slowly": [0.2, 0, 0, 1.05],
    "move backward slowly": [-0.2, 0, 0, 1.05],
    "turn left slowly": [0, 0, 0.2, 1.05],
    "turn right slowly": [0, 0, -0.2, 1.05],
    "move to the left slowly": [0, 0.2, 0, 1.05],
    "move to the right slowly": [0, -0.2, 0, 1.05],
    "stop": [0, 0, 0, 1.05],
    "stay static": [0, 0, 0, 1.05],
    "stand up": [0, 0, 0, 1.05],
    "crouch down fully": [0, 0, 0, 0.45],
    "crouch down": [0, 0, 0, 0.45],
    "crouch down half": [0, 0, 0, 0.75],
    "Pick up things on the ground": [0, 0, 0, 0.45],
    "Pick up things on the sofa": [0, 0, 0, 0.75]
}

DEFAULT_QPOS = np.float32([
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])

DEFAULT_ROOT_POSE = np.float32([
    0., 0., 0.85,  # position [m]
    1., 0., 0., 0.  # qua
])

# v1
KPs = np.float32([
    100, 100, 100, 200, 80, 20,
    100, 100, 100, 200, 80, 20,
    300, 300, 300,
    90, 60, 20, 60, 20, 20, 20,
    90, 60, 20, 60, 20, 20, 20,
])

KDs = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])
