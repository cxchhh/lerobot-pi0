DATA_PATH=sim-v4.0-MoveGrasp.zip
~/ads-cli cp ~/sandbox/wholebody_grasp/lerobot_dataset/$DATA_PATH  $aoss/cxc/datasets/$DATA_PATH

ssh ksyun "~/ads-cli cp $aoss/cxc/datasets/$DATA_PATH /mnt/kpfs/chenxuchuan/sandbox/G1-VLA/$DATA_PATH"
