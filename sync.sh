CKPT_PATH=outputs/train/2025-11-27/15-53-52_sim-v4.7-MoveGrasp/checkpoints/last/pretrained_model
DATA_PATH=sim-v4.7-MoveGrasp

mkdir -p $CKPT_PATH
ssh ksyun "~/ads-cli cp /mnt/kpfs/chenxuchuan/sandbox/lerobot/$CKPT_PATH $aoss/cxc/pi0/ckpt/$DATA_PATH"
~/ads-cli cp $aoss_in/cxc/pi0/ckpt/$DATA_PATH $CKPT_PATH 

mkdir -p ./lerobot_data/$DATA_PATH/meta
ssh ksyun "~/ads-cli cp /mnt/kpfs/danshili/Workspace/lerobot/storage/data/lerobot/$DATA_PATH/meta $aoss/cxc/pi0/meta"
~/ads-cli cp $aoss_in/cxc/pi0/meta ./lerobot_data/$DATA_PATH/meta

echo -e "TOKENIZERS_PARALLELISM=false python server.py \\" > run_server.sh 
echo -e "   --policy.path=$CKPT_PATH \\" >> run_server.sh
echo -e "   --dataset.repo_id=lerobot_data/$DATA_PATH \\" >> run_server.sh
echo -e "   --dataset.root=./lerobot_data/$DATA_PATH" >> run_server.sh
