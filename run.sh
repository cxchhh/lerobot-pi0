NUM_GPUS=4

torchrun \
  --nproc_per_node=$NUM_GPUS \
  lerobot/scripts/train_ddp.py \
  --dataset.repo_id=lerobot/grasp_ycb_box_absolute \
  --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/grasp_ycb_box_absolute \
  --wandb.enable=true \
  --batch_size=16 \
  --steps=80000 \
  --policy.path=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  # --resume=true \
  # --config_path=outputs/train/2025-06-30/12-43-37_pi0/checkpoints/last/pretrained_model \
 
  


