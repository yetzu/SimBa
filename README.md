conda config --add channels defaults
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
conda config --show channels

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

conda create -n metai python=3.11
conda activate metai

conda install gxx_linux-64 cuda-toolkit=12.1 -y
conda install pandas matplotlib scipy opencv PyYAML seaborn pydantic cartopy rasterio -y
pip install torch torchvision torchaudio tensorboard timm pytorch-msssim lpips
pip install lightning lightning-utilities 

mkdir mamba_build_temp
cd mamba_build_temp
tar -xzf /home/dataset-assist-0/code/mamba_ssm-2.2.6.post3.tar.gz
cd mamba_ssm-2.2.6.post3/
export FORCE_BUILD=1
python setup.py install

pip install mamba-ssm


find /home/dataset-assist-1/SevereWeather_AI_2025/CP/TestSetB -maxdepth 1 -mindepth 1 -type d | xargs -I {} -P 32   rsync -aW --ignore-existing {} ./TestSetB

nohup bash run.scwds.mamba.sh train > train_mamba_scwds.log 2>&1 &

pip install deepspeed