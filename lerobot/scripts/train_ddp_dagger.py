#!/usr/bin/env python
import itertools
import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from torch.utils.data import random_split, DataLoader
import tqdm
import swanlab

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
swanlab.sync_wandb(wandb_run=False)

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir, get_step_identifier, load_training_state,
    save_checkpoint, update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number, init_logging, has_method
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from wholebody_grasp.scripts.v8_to_lerobot import VecRunner, _parse_visible_gpus

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, grad_scaler, lr_scheduler=None, use_amp=False):
    start_time = time.perf_counter()
    policy.train()
    device = next(policy.parameters()).device

    with torch.autocast(device_type="cuda") if use_amp else nullcontext():
        loss, output_dict = policy.module.forward(batch)  # note: policy is DDP
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy.module, "update"):
        policy.module.update()

    with torch.no_grad():
        action_pred = policy.module.get_action_chunk(batch)
        # print(action_pred.shape)
        output_dict["action"] = action_pred.squeeze(1).cpu().numpy()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]['lr']
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def sanitize_for_wandb(log_dict):
    safe_dict = {}
    for k, v in log_dict.items():
        try:
            if isinstance(v, (int, float, bool, str)):
                safe_dict[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                safe_dict[k] = v.item()
            elif isinstance(v, torch.Tensor):
                continue
            elif isinstance(v, (list, np.ndarray)) and np.array(v).ndim <= 2:
                safe_dict[k] = vars
        except:
            pass
    return safe_dict

def eval_on_val_dataset(cfg, model, val_dataset, device, max_batches=10):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False
        )

    limited_loader = itertools.islice(val_loader, max_batches)
    for batch in tqdm.tqdm(limited_loader, total=max_batches):
        batch={
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        with torch.no_grad():
            loss, output_dict = model.forward(batch) 

        total_loss += loss.item()
        total_samples += 1

    avg_loss = total_loss / total_samples
    return {"val_loss": avg_loss}

def train(rank: int, world_size: int, cfg: TrainPipelineConfig):
    setup_ddp(rank, world_size)
    if rank == 0:
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed + rank)
        generator = torch.Generator().manual_seed(cfg.seed + rank)
    else:
        generator = None

    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg.policy.device = f"cuda:{rank}"

    ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
    ds_meta.stats['observation.state']["mean"] = np.zeros(ds_meta.features['observation.state']["shape"])
    ds_meta.stats['observation.state']["std"] = np.ones(ds_meta.features['observation.state']["shape"])
    
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and rank == 0:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    gpu_id = _parse_visible_gpus()[rank]
    env = VecRunner(num_envs=cfg.batch_size, gpu_ids=[gpu_id])
    batch, dones = env.reset()
    print(rank, "env starts")
    
    print(rank, "load policy into", device)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    policy.to(device)
    policy = DDP(policy, device_ids=[rank])

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy.module)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    step = 0

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    if rank == 0:
        wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project else None
    else:
        wandb_logger = None

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size * world_size, 1, 1, train_metrics, initial_step=step)

    for _ in range(step, cfg.steps):
        # breakpoint()
        start_time = time.perf_counter()
       
        train_tracker.dataloading_s = time.perf_counter() - start_time

        batch["action"] = env.get_teacher_actions()[:, None, ...] # 1 step
        batch["task"] = ["reaching the object in the front"] * cfg.batch_size
        for key in batch:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.tensor(batch[key]).to(device, non_blocking=True)
                if "observation.images" in key:
                    batch[key] = batch[key].permute(0,3,1,2).float() / 255.
                if "action" in key:
                    batch[key] = batch[key].float()
                # print(key, batch[key].shape, batch[key].dtype)

        train_tracker, output_dict = update_policy(
            train_tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler, lr_scheduler=lr_scheduler, use_amp=cfg.policy.use_amp)
        # print(output_dict.keys())
        # exit(0)
        step += 1
        train_tracker.step()
        batch, dones = env.step(output_dict["action"])

        if rank == 0:
            if cfg.log_freq > 0 and step % cfg.log_freq == 0:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    wandb_log_dict.update(sanitize_for_wandb(output_dict))
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy.module, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)


    if rank == 0 and eval_env:
        eval_env.close()
    cleanup_ddp()
    if rank == 0:
        logging.info("End of training")
        
@parser.wrap()
def main(cfg: TrainPipelineConfig):
    cfg.validate()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size, cfg)

if __name__ == "__main__":
    init_logging()
    main()
