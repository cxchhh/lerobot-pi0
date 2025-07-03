TOKENIZERS_PARALLELISM=false python genesis_eval/inference_single.py \
    --policy.path=outputs/train/2025-06-29/03-01-02_pi0/checkpoints/030000/pretrained_model \
    --dataset.repo_id=lerobot/example_repo_overfit_delta \
    --dataset.root=/mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/example_repo_overfit_delta \