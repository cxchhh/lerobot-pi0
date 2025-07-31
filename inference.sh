TOKENIZERS_PARALLELISM=false python genesis_eval/inference_single.py \
    --policy.path=outputs/train/2025-07-27/00-51-40_grasp_ycb_box_broader/checkpoints/120000/pretrained_model \
    --dataset.repo_id=lerobot/grasp_ycb_box_broader \
    --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/grasp_ycb_box_broader