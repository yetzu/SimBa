# run/train_scwds_mamba.py

"""
MetMamba 模型训练入口脚本 (Training Script).

本脚本基于 PyTorch Lightning CLI (Command Line Interface) 构建，作为 MetMamba 模型
在 SCWDS (短时强对流天气) 数据集上的标准训练入口。

主要功能：
    1. 环境初始化：配置系统路径与 Tensor 运算精度。
    2. CLI 封装：通过 LightningCLI 自动解析命令行参数与配置文件 (YAML)。
    3. 训练流程：自动实例化模型 (MetMambaTrainer) 与数据模块 (ScwdsDataModule)，并启动训练循环。

Usage:
    # 使用默认配置运行
    python run/train_scwds_mamba.py fit --config metai/config.yaml

    # 覆盖特定参数 (例如修改 Batch Size 和 Epochs)
    python run/train_scwds_mamba.py fit --config metai/config.yaml \
        --data.batch_size=8 --trainer.max_epochs=50
"""

import sys
import os
import torch

# 当前脚本位于 code/run/ 目录下
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning.pytorch.cli import LightningCLI

# 导入自定义的数据模块 (DataModule) 和 模型模块 (LightningModule)
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.mamba import MetMambaTrainer as MeteoMambaModule 

def main():
    """
    训练主函数。

    配置计算精度并启动 Lightning CLI 进行模型训练。
    """
    # 设置矩阵乘法运算精度为 'high'
    # 在 Ampere 架构及更新的 GPU 上，这将启用 TensorFloat-32 (TF32)，
    # 在保持足够精度的前提下显著提升 FP32 矩阵乘法性能。
    torch.set_float32_matmul_precision('high')

    # 初始化 LightningCLI
    # 这一步会自动执行以下操作：
    # 1. 解析命令行参数和配置文件 (基于 OmegaConf)。
    # 2. 实例化 Model (MeteoMambaModule) 和 DataModule (ScwdsDataModule)。
    # 3. 配置 Trainer (包括 GPU、Checkpoint、Logger 等)。
    # 4. 执行 trainer.fit() (因为 run=True)。
    cli = LightningCLI(
        model_class=MeteoMambaModule,
        datamodule_class=ScwdsDataModule,
        save_config_callback=None,  # 不强制保存解析后的配置到文件
        run=True,                   # 初始化后立即运行 fit/validate/test
        parser_kwargs={"parser_mode": "omegaconf"} 
    )

if __name__ == "__main__":
    main()