CKPT_PATH=outputs/train/2026-06-29/17-09-50_bfm-v1.9/checkpoints/050000/pretrained_model
DATA_PATH=bfm-v1.9

mkdir -p $CKPT_PATH
# ssh ksyun "~/ads-cli cp /mnt/kpfs/chenxuchuan/sandbox/lerobot/$CKPT_PATH $aoss/cxc/pi0/ckpt/$DATA_PATH"
# ~/ads-cli cp $aoss_in/cxc/pi0/ckpt/$DATA_PATH $CKPT_PATH 
rsync -avP --info=progress2 --partial-dir=.rsync-partial --blocking-io ksyun:/mnt/kpfs/chenxuchuan/sandbox/lerobot/$CKPT_PATH $CKPT_PATH/..

mkdir -p ./lerobot_data/$DATA_PATH/meta
# ssh ksyun "~/ads-cli cp /mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/$DATA_PATH/meta $aoss/cxc/pi0/meta"
# ~/ads-cli cp $aoss_in/cxc/pi0/meta ./lerobot_data/$DATA_PATH/meta
rsync -avP --info=progress2 --partial-dir=.rsync-partial --blocking-io ksyun:/mnt/kpfs/chenxuchuan/sandbox/G1-VLA/$DATA_PATH/meta ./lerobot_data/$DATA_PATH

echo -e "TOKENIZERS_PARALLELISM=false python server.py \\" > run_server.sh 
echo -e "   --policy.path=$CKPT_PATH \\" >> run_server.sh
echo -e "   --dataset.repo_id=lerobot_data/$DATA_PATH \\" >> run_server.sh
echo -e "   --dataset.root=./lerobot_data/$DATA_PATH" >> run_server.sh
