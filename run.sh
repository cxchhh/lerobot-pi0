NUM_GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT="29503"
export ID=grasp_ycb_box_fixurdf

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  lerobot/scripts/train_ddp.py \
  --dataset.repo_id=lerobot/$ID \
  --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/$ID \
  --job_name=$ID \
  --wandb.enable=true \
  --batch_size=16 \
  --steps=120000 \
  --policy.path=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  # --resume=true \
  # --config_path=outputs/train/2025-07-18/12-42-46_grasp_ycb_box_absolute_textured_small4/checkpoints/050000/pretrained_model \
  
 
  
    
 
  


