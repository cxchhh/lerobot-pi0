import multiprocessing
from rich.progress import track
import os
GX_STORAGE_PATH = os.getenv("GX_STORAGE_PATH")
from pathlib import Path
from gx_utils.dtype import Render, ListRender, Physics, ListPhysics, Scene, ListScene, ID, ListPlan, Box, Mesh
import argparse
from tqdm import tqdm

def evaluate(arg_param):
    gpu, tasks = arg_param
    root = None # 
    date = '2025.06.08'
    names = ['fast_DDPM','fast_obs2','fast','fast']

    name = names[gpu]
    delta = int('delta' in name)
    for i in range(len(tasks)):
        uid, ii = tasks[i]
        command = f'CUDA_VISIBLE_DEVICES={gpu} python eval.py --checkpoint {root}/{date}/{name}/checkpoints/latest.ckpt --output_dir data/{date}_G1_VLA_{name} --uid {uid} --ii {ii} --delta_closeness {delta}'
        os.system(command)



# load evaluation scenes
test_split_root = f'{GX_STORAGE_PATH}/sim'
all_traj_indices = []
uids = sorted(os.listdir(Path(test_split_root) / 'render' / 'debug_chomp'))
parser = argparse.ArgumentParser(description="Sample layout")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

gpu = args.gpu
tasks = []
uid_list = uids[(gpu)*100:] + uids[:(gpu)*100]
for uid in tqdm(uid_list):
    try:
        physicss = ListPhysics.load_h5_list(Path(test_split_root) / 'physics' / 'debug_chomp' / uid / '0' / 'physics.h5')
        for i in range(len(physicss)):
            tasks.append([uid,i])
            all_traj_indices.append([gpu, tasks])
    except:
        pass

with multiprocessing.Pool(1) as pool:
    it = track(
        pool.imap_unordered(evaluate, all_traj_indices), 
        total=len(all_traj_indices), 
        description='evaluating', 
    )
    list(it)