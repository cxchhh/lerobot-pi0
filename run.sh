NUM_GPUS=1

torchrun \
  --nproc_per_node=$NUM_GPUS \
  lerobot/scripts/train_ddp.py \
  --dataset.repo_id=lerobot/example_repo_0 \
  --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/example_repo_0 \
  --wandb.enable=true \
  --policy.path=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  # --resume=true \
  # --config_path=outputs/train/2025-06-13/17-10-23_pi0/checkpoints/020000/pretrained_model/ \


