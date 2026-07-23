CKPT_PATH=outputs/train/2026-07-22/20-29-00_bfm-v2.11/checkpoints/080000/pretrained_model
DATA_PATH=bfm-v2.11

mkdir -p $CKPT_PATH
# ssh ksyun "~/ads-cli cp /mnt/kpfs/chenxuchuan/sandbox/lerobot/$CKPT_PATH $aoss/cxc/pi0/ckpt/$DATA_PATH"
# ~/ads-cli cp $aoss_in/cxc/pi0/ckpt/$DATA_PATH $CKPT_PATH 
rsync -avP --info=progress2 --partial-dir=.rsync-partial --blocking-io 10.0.1.12:/mnt/kpfs/chenxuchuan/sandbox/lerobot/$CKPT_PATH $CKPT_PATH/..

mkdir -p ./lerobot_data/$DATA_PATH/meta
# ssh ksyun "~/ads-cli cp /mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/$DATA_PATH/meta $aoss/cxc/pi0/meta"
# ~/ads-cli cp $aoss_in/cxc/pi0/meta ./lerobot_data/$DATA_PATH/meta
rsync -avP --info=progress2 --partial-dir=.rsync-partial --blocking-io 10.0.1.12:/mnt/kpfs/chenxuchuan/sandbox/G1-VLA/$DATA_PATH/meta ./lerobot_data/$DATA_PATH

echo -e "SAVE_ATTN=1 TOKENIZERS_PARALLELISM=false python server.py \\" > run_server.sh
echo -e "   --policy.path=$CKPT_PATH \\" >> run_server.sh
echo -e "   --dataset.repo_id=lerobot_data/$DATA_PATH \\" >> run_server.sh
echo -e "   --dataset.root=./lerobot_data/$DATA_PATH \"\$@\"" >> run_server.sh
