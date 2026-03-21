NUM_GPUS=1
export MASTER_ADDR="localhost"
export MASTER_PORT="29514"
export ID=sim-v9.36
export RANK=0
export WORLD_SIZE=1

python lerobot/scripts/train_ddp.py \
  --dataset.repo_id=G1-VLA/$ID \
  --dataset.root=/mnt/kpfs/chenxuchuan/sandbox/G1-VLA/$ID \
  --job_name=$ID \
  --batch_size=16 \
  --wandb.enable=false \
  --steps=50000 \
  --policy.path=/mnt/kpfs/chenxuchuan/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch_plan \
  # --resume=true \
  # --config_path=outputs/train/2026-03-20/20-50-37_sim-v9.35/checkpoints/050000/pretrained_model
