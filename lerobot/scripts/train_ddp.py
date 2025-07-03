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
from lerobot.scripts.eval import eval_policy

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
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

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]['lr']
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def sanitize_for_wandb(log_dict):
    safe_dict = {}
    for k, v in log_dict.items():
        try:
            # 尝试只保留 wandb 支持的基本类型
            if isinstance(v, (int, float, bool, str)):
                safe_dict[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                safe_dict[k] = v.item()
            elif isinstance(v, torch.Tensor):
                continue  # 跳过多维 Tensor，避免 warning
            elif isinstance(v, (list, np.ndarray)) and np.array(v).ndim <= 2:
                safe_dict[k] = v
            # 你也可以加其他允许类型判断
        except:
            pass  # 忽略不支持的
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

    dataset = make_dataset(cfg)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)
    
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and rank == 0:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    print(rank, "make policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.to(device)
    policy = DDP(policy, device_ids=[rank])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        pin_memory=True,
        drop_last=False
    )
    dl_iter = cycle(train_loader)

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
    train_tracker = MetricsTracker(cfg.batch_size * world_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step)

    for _ in range(step, cfg.steps):
        # breakpoint()
        train_sampler.set_epoch(step)
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler, lr_scheduler=lr_scheduler, use_amp=cfg.policy.use_amp)

        step += 1
        train_tracker.step()

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

            if cfg.eval_freq > 0 and step % cfg.eval_freq == 0:
                # step_id = get_step_identifier(step, cfg.steps)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                        eval_info = eval_on_val_dataset(cfg, policy.module, val_dataset, device)

                logging.info(f"[Eval] Step {step}: val_loss = {eval_info['val_loss']:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict(eval_info, step, mode="eval")

                # eval_metrics = {
                #     "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                #     "pc_success": AverageMeter("success", ":.1f"),
                #     "eval_s": AverageMeter("eval_s", ":.3f"),
                # }
                # eval_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step)
                # eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                # eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                # eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                # logging.info(eval_tracker)
                # if wandb_logger:
                #     wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                #     wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                #     wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

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
