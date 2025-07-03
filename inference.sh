TOKENIZERS_PARALLELISM=false python genesis_eval/inference_single.py \
    --policy.path=outputs/train/2025-06-27/01-56-24_pi0/checkpoints/045000/pretrained_model \
    --dataset.repo_id=lerobot/example_repo_0 \
    --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/example_repo_0/