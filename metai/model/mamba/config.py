# metai/model/mamba/config.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple, Union, List, Optional

class ModelConfig(BaseModel):
    """
    MetMamba 模型配置类。

    用于定义 MetMamba 短时降水预测模型的所有超参数，包括输入输出规格、网络架构细节、
    训练优化策略以及多尺度损失函数的权重配置。

    Attributes:
        model_config (ConfigDict): Pydantic 配置字典，允许任意类型并禁止受保护命名空间。
    """

    # =========================================================================
    # 1. 基础环境与路径配置 (Basic Environment & Paths)
    # =========================================================================
    model_name: str = Field(
        default="met_mamba", 
        description="模型实验名称。该名称将作为日志文件、Checkpoint 权重文件以及可视化结果的唯一标识前缀。"
    )
    data_path: str = Field(
        default="data/samples.jsonl", 
        description="数据集索引文件路径（.jsonl 格式）。每一行应包含指向具体数据样本（如 .npz 或 .h5）的元数据。"
    )
    save_dir: str = Field(
        default="./output/mamba", 
        description="实验输出目录。用于存放训练过程中的日志 (TensorBoard/W&B)、保存的模型权重及验证集预测结果。"
    )
    
    # =========================================================================
    # 2. 输入输出形状配置 (Input/Output Shapes)
    # =========================================================================
    in_shape: Tuple[int, int, int, int] = Field(
        default=(10, 54, 256, 256), 
        description=(
            "输入张量的几何形状，格式为 [T_in, C_in, H, W]。\n"
            "物理含义：\n"
            "  - T_in (10): 输入历史帧数（过去1小时，6分钟/帧）；\n"
            "  - C_in (54): 输入通道总数（包含多高度层雷达回波、数值模式预报变量及静态地理信息）；\n"
            "  - H, W (256): 空间分辨率。"
        )
    )
    out_seq_length: int = Field(
        default=20, 
        description="预测序列长度 T_out。表示模型需输出的未来帧数（未来2小时，6分钟/帧）。"
    )
    out_channels: int = Field(
        default=1, 
        description="输出通道数 C_out。通常为 1，即预测的组合反射率因子（Composite Reflectivity）。"
    )

    # =========================================================================
    # 3. 数据加载配置 (Dataloader)
    # =========================================================================
    batch_size: int = Field(
        default=4, 
        description="单张 GPU 卡的批次大小 (Batch Size)。显存允许的情况下建议调大以稳定梯度。"
    )
    num_workers: int = Field(
        default=4, 
        description="PyTorch DataLoader 的子进程数量。用于 CPU 端的数据预处理与加载加速。"
    )
    seed: int = Field(
        default=42, 
        description="全局随机种子。固定该值可确保模型初始化、数据打乱顺序的一致性，保证实验可复现。"
    )
    
    # =========================================================================
    # 4. 模型结构超参数 (Model Architecture)
    # =========================================================================
    # 4.1 空间编码器/解码器 (Encoder/Decoder)
    hid_S: int = Field(
        default=64, 
        description="空间特征提取网络的隐含层通道数 (Spatial Hidden Dimension)。决定了 CNN 提取空间特征的容量。"
    )
    N_S: int = Field(
        default=4, 
        description="空间编码器和解码器的堆叠层数。层数越深，感受野越大，提取的语义特征越高级。"
    )
    spatio_kernel_enc: int = Field(
        default=3, 
        description="编码器 (Encoder) 中的卷积核大小 (Kernel Size)。"
    )
    spatio_kernel_dec: int = Field(
        default=3, 
        description="解码器 (Decoder) 中的卷积核大小 (Kernel Size)。"
    )
    
    # 4.2 时序演变模块 (Temporal Translator - Mamba)
    hid_T: int = Field(
        default=256, 
        description="时序演变模块 (MidNet) 的隐含层通道数。即 Mamba 模块处理的特征维度。"
    )
    N_T: int = Field(
        default=8, 
        description="Mamba 模块的堆叠层数。决定了模型捕获长时序依赖关系的能力。"
    )
    mlp_ratio: float = Field(
        default=4.0, 
        description="Feed-Forward Network (FFN) 的扩展比例。内部维度 = hid_T * mlp_ratio。"
    )
    drop: float = Field(
        default=0.0, 
        description="Dropout 概率。用于防止全连接层过拟合。"
    )
    drop_path: float = Field(
        default=0.0, 
        description="Stochastic Depth (Drop Path) 概率。用于深层残差网络的正则化，随机丢弃部分残差分支。"
    )

    d_state: int = Field(
        default=16, 
        description="SSM (State Space Model) 的潜在状态维度 (State Dimension)。控制状态空间模型对历史信息的记忆容量。"
    )

    d_conv: int = Field(
        default=4, 
        description="SSM 模块内部的局部一维卷积核大小。用于在计算 SSM 之前捕获局部时序特征。"
    )
    
    expand: int = Field(
        default=2, 
        description="Mamba 块内部的维度扩展倍数 (Expansion Factor)。输入特征会先被投影到 expand * hid_T 维度。"
    )

    # =========================================================================
    # 5. 训练与优化配置 (Training & Optimization)
    # =========================================================================
    max_epochs: int = Field(
        default=100, 
        description="最大训练轮次 (Epochs)。"
    )
    opt: str = Field(
        default="adamw", 
        description="优化器类型选择。支持 'adamw', 'sgd' 等标准 PyTorch 优化器。"
    )
    sched: str = Field(
        default="cosine", 
        description="学习率调度策略 (LR Scheduler)。如 'cosine' (余弦退火), 'one_cycle' 等。"
    )
    lr: float = Field(
        default=1e-3, 
        description="初始学习率 (Learning Rate)。"
    )
    min_lr: float = Field(
        default=1e-5, 
        description="最小学习率。用于 Cosine Annealing 策略中的下界。"
    )
    warmup_epoch: int = Field(
        default=5, 
        description="学习率预热轮数 (Warmup Epochs)。在此期间学习率从 0 线性增加到 lr，有助于稳定训练初期梯度。"
    )
    weight_decay: float = Field(
        default=1e-2, 
        description="权重衰减系数 (Weight Decay)。对应 L2 正则化，用于防止模型过拟合。"
    )

    # =========================================================================
    # 6. 损失函数权重 (Loss Weights)
    # =========================================================================
    loss_weight_l1: float = Field(
        default=1.0, 
        description="L1 Loss (MAE) 的权重。关注逐像素的数值准确性。"
    )
    loss_weight_ssim: float = Field(
        default=0.5, 
        description="MS-SSIM (多尺度结构相似性) 损失的权重。关注图像的结构信息和感知质量，防止预测模糊。"
    )
    loss_weight_csi: float = Field(
        default=1.0, 
        description="Soft-CSI (临界成功指数) 损失的权重。针对气象降水任务优化，关注强回波（暴雨）区域的命中率。"
    )
    loss_weight_spectral: float = Field(
        default=0.1, 
        description="频域距离损失 (Spectral Loss) 的权重。通过约束功率谱密度，减轻长时预测中的图像模糊问题。"
    )
    loss_weight_evo: float = Field(
        default=0.5, 
        description="演变一致性损失 (Evolution Consistency Loss) 的权重。约束特征空间中的时序演变平滑度。"
    )
    use_curriculum_learning: bool = Field(
        default=True, 
        description="是否启用课程学习 (Curriculum Learning)。若启用，训练初期损失权重可能动态调整，先易后难。"
    )

    # =========================================================================
    # 7. 系统与训练器杂项 (System & Trainer Misc)
    # =========================================================================
    precision: str = Field(
        default="16-mixed", 
        description="训练计算精度。'16-mixed' 表示混合精度训练 (AMP)，可节省显存并加速；'32' 表示全精度 (FP32)。"
    )
    accelerator: str = Field(
        default="auto", 
        description="硬件加速器类型。可选 'auto' (自动检测), 'gpu', 'cpu', 'tpu'。"
    )
    devices: Union[int, str, List[int]] = Field(
        default="auto", 
        description="设备编号配置。例如 [0, 1] 表示使用前两张 GPU，'auto' 表示自动选择。"
    )
    check_val_every_n_epoch: int = Field(
        default=1, 
        description="验证频率。每间隔多少个 Epoch 执行一次验证集评估。"
    )
    
    # 早停配置 (Early Stopping)
    early_stop_monitor: str = Field(
        default="val_score", 
        description="早停 (Early Stopping) 的监控指标名称。"
    )
    early_stop_mode: str = Field(
        default="max", 
        description="早停监控指标的模式。'max' 表示指标越高越好 (如 CSI)，'min' 表示指标越低越好 (如 Loss)。"
    )
    early_stop_patience: int = Field(
        default=20, 
        description="早停忍耐轮数 (Patience)。若指标在连续 N 轮内未改善，则停止训练。"
    )

    # =========================================================================
    # 8. 辅助属性与方法 (Properties & Methods)
    # =========================================================================
    
    @property
    def in_seq_length(self) -> int:
        """
        获取输入序列的时间维度长度 T_in。

        Returns:
            int: 输入帧数 (例如 10)。
        """
        return self.in_shape[0]

    @property
    def channels(self) -> int:
        """
        获取输入的特征通道总数 C_in。

        Returns:
            int: 通道数 (例如 54 = 雷达 + NWP + GIS)。
        """
        return self.in_shape[1]
    
    @property
    def resize_shape(self) -> Tuple[int, int]:
        """
        获取目标图像的空间分辨率 (H, W)。

        Returns:
            Tuple[int, int]: 图像的高度和宽度 (例如 (256, 256))。
        """
        return (self.in_shape[2], self.in_shape[3])

    def to_dict(self) -> dict:
        """
        将配置对象转换为字典格式，并补充推导出的辅助属性。

        Returns:
            dict: 包含所有配置项及 'in_seq_length', 'channels', 'resize_shape' 的字典。
        """
        data = self.model_dump()
        data['in_seq_length'] = self.in_seq_length
        data['channels'] = self.channels
        data['resize_shape'] = self.resize_shape
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        """
        从字典数据创建 ModelConfig 实例。

        Args:
            data (dict): 包含配置参数的字典。

        Returns:
            ModelConfig: 初始化后的配置对象。
        """
        return cls(**data)

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())