TOKENIZERS_PARALLELISM=false python genesis_eval/inference_single.py \
    --policy.path=outputs/train/2025-07-20/13-27-17_grasp_ycb_box_fixurdf/checkpoints/120000/pretrained_model \
    --dataset.repo_id=lerobot/grasp_ycb_box_fixurdf \
    --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/grasp_ycb_box_fixurdf