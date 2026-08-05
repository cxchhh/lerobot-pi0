CKPT_PATH=outputs/train/2026-08-04/22-32-49_teleop-v1.4-pi05/checkpoints/020000/pretrained_model
DATA_PATH=teleop-v1.4

mkdir -p $CKPT_PATH
# ssh ksyun "~/ads-cli cp /mnt/kpfs/chenxuchuan/sandbox/lerobot/$CKPT_PATH $aoss/cxc/pi0/ckpt/$DATA_PATH"
# ~/ads-cli cp $aoss_in/cxc/pi0/ckpt/$DATA_PATH $CKPT_PATH 
rsync -avP --info=progress2 --partial-dir=.rsync-partial --blocking-io aliyun:/mnt/home/chenxuchuan/sandbox/lerobot/$CKPT_PATH/ $CKPT_PATH/

mkdir -p ./lerobot_data/$DATA_PATH/meta
# ssh ksyun "~/ads-cli cp /mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/$DATA_PATH/meta $aoss/cxc/pi0/meta"
# ~/ads-cli cp $aoss_in/cxc/pi0/meta ./lerobot_data/$DATA_PATH/meta
rsync -avP --info=progress2 --partial-dir=.rsync-partial --blocking-io aliyun:/mnt/home/chenxuchuan/sandbox/lerobot/lerobot_dataset/$DATA_PATH/meta ./lerobot_data/$DATA_PATH

echo -e "SAVE_ATTN=1 TOKENIZERS_PARALLELISM=false \\" > run_server.sh
echo -e "systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=2G \\" >> run_server.sh
echo -e "   python server.py \\" >> run_server.sh
echo -e "   --policy.path=$CKPT_PATH \\" >> run_server.sh
echo -e "   --dataset.repo_id=lerobot_data/$DATA_PATH \\" >> run_server.sh
echo -e "   --dataset.root=./lerobot_data/$DATA_PATH \"\$@\"" >> run_server.sh
