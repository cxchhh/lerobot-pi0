NUM_GPUS=4

torchrun \
  --nproc_per_node=$NUM_GPUS \
  lerobot/scripts/train_ddp.py \
  --dataset.repo_id=unitree/example_repo_1 \
  --dataset.root=$HOME/.cache/huggingface/lerobot/unitree/example_repo_1 \
  --wandb.enable=true \
  --policy.path=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  # --resume=true \
  # --config_path=outputs/train/2025-06-13/17-10-23_pi0/checkpoints/020000/pretrained_model/ \


