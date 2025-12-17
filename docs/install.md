
# 开发环境搭建

## 依赖说明
本项目基于 Python 3.11 和 PyTorch 构建，推荐使用 Linux 环境并配合 CUDA 12.1 进行加速。

## 核心依赖
* **Python**: 3.11
* **CUDA Toolkit**: 12.1
* **深度学习框架**:
    * `torch`, `torchvision`, `torchaudio`
    * `lightning` (PyTorch Lightning)
    * `deepspeed`
    * `timm`
    * `mamba-ssm` (需编译)
* **科学计算与图像处理**:
    * `pandas`, `scipy`, `matplotlib`, `seaborn`
    * `opencv`
    * `rasterio`, `cartopy`
* **工具库**:
    * `jsonargparse`, `PyYAML`, `pytest`

## 安装命令

```bash
# 创建环境
conda create -n metai python=3.11
conda activate metai

# 安装系统级依赖
conda install gxx_linux-64 cuda-toolkit=12.1 -y
conda install pytest pandas matplotlib scipy opencv PyYAML seaborn pydantic cartopy rasterio -y

# 安装 Python 库
pip install torch torchvision torchaudio tensorboard timm pytorch-msssim lpips deepspeed
pip install lightning lightning-utilities 
pip install -U "jsonargparse[signatures]>=4.18.0"

# 安装 mamba-ssm 库
# pip install mamba-ssm==2.2.6.post3

# 竞赛服务器无法直接安装mamba_ssm，需编译
mkdir mamba_build_temp
cd mamba_build_temp
# 解压源码包 (路径需根据实际情况调整)
tar -xzf mamba_ssm-2.2.6.post3.tar.gz
cd mamba_ssm-2.2.6.post3/
export FORCE_BUILD=1
python setup.py install
pip install mamba-ssm
```