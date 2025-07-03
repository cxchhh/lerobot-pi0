NUM_GPUS=2

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=29501 \
  lerobot/scripts/train_ddp.py \
  --dataset.repo_id=lerobot/teleop-v1.0-PickNaiLong-InitDown \
  --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/teleop-v1.0-PickNaiLong-InitDown \
  --wandb.enable=true \
  --policy.path=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  --steps=10000 \
  # --resume=true \
  # --config_path=outputs/train/2025-06-13/17-10-23_pi0/checkpoints/020000/pretrained_model/ \
  


