TOKENIZERS_PARALLELISM=false python eval.py \
    --policy.path=outputs/train/2025-06-17/02-20-39_pi0/checkpoints/020000/pretrained_model/ \
    --dataset.repo_id=unitree/example_repo_1 \
    --dataset.root=$HOME/.cache/huggingface/lerobot/unitree/example_repo_1 \