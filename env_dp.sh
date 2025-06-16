conda init bash
bash
export GX_STORAGE_PATH=/mnt/afs/danshili/GalbotVLA/galbot_vla/src/sim_data_gen/storage;
export WANDB_API_KEY=a77536ca9e8c8a995fa62f0eaea2c06b5236b845;
export CMAKE_PREFIX_PATH=/mnt/afs/danshili/Dependencies/cmake:/mnt/afs/danshili/Dependencies/pybind11/share/cmake:$CMAKE_PREFIX_PATH;
export ACCEPT_EULA=1
cd /mnt/afs/danshili/Workspace/diffusion_policy
pip install -e /mnt/afs/danshili/GalbotVLA/galbot_vla/src/sim_data_gen;
pip install -e /mnt/afs/danshili/GalbotVLA/gx_sim;
pip install -e /mnt/afs/danshili/GalbotVLA/gx_utils;
pip install rich;
pip install numba;
pip install cffi==1.15.1;
pip install ipykernel==6.16;
pip install zarr==2.12.0;
pip install numcodecs==0.10.2;
pip install hydra-core==1.2.0;
pip install dill==0.3.5.1;
pip install pymunk==6.2.1;
pip install threadpoolctl==3.1.0;
pip install shapely==1.8.4;
pip install psutil==5.9.2;
pip install click==8.0.4;
pip install diffusers==0.11.1;
pip install av==10.0.0;
pip install huggingface_hub==0.25.2;
pip install pandas
pip install -e /mnt/afs/danshili/Workspace/diffusion_policy;
pip install numpy==1.26.4
pip install numba --upgrade
