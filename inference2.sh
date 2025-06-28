TOKENIZERS_PARALLELISM=false python genesis_eval/inference_single.py \
    --policy.path=outputs/train/2025-06-27/18-48-54_pi0/checkpoints/050000/pretrained_model \
    --dataset.repo_id=lerobot/example_repo_absoluteqpos_1 \
    --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/example_repo_absoluteqpos_1 \