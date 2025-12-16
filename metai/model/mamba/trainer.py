# metai/model/mamba/trainer.py

import torch
import torch.nn.functional as F
import lightning as l
from typing import Any, Dict, cast, Tuple, Optional
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from metai.model.core import get_optim_scheduler
from .model import MetMamba
from .loss import HybridLoss
from .metrices import MetScore

class MetMambaTrainer(l.LightningModule):
    """
    MetMamba Lightning 训练器 (MetMamba Lightning Trainer).

    该类封装了 MetMamba 模型的训练、验证、测试和推理逻辑。
    它集成了课程学习 (Curriculum Learning) 策略，用于动态调整不同损失函数的权重，
    以及针对气象降水任务的评估指标计算。

    Args:
        in_shape (Tuple[int, int, int, int]): 输入数据的形状 (T, C, H, W)。
        out_seq_length (int): 预测序列的长度 (T_out)。默认值: 20。
        out_channels (int): 输出通道数。默认值: 1 (通常为降水量)。
        
        # --- 模型架构参数 (Model Architecture) ---
        hid_S (int): 空间编码器的隐藏层通道数。默认值: 128。
        hid_T (int): 时空转换模块 (MidNet) 的通道数。默认值: 512。
        N_S (int): 空间编码器/解码器的层数。默认值: 4。
        N_T (int): 时空转换模块的堆叠层数。默认值: 12。
        mlp_ratio (float): MLP 层的扩展比例。默认值: 4.0。
        spatio_kernel_enc (int): 编码器卷积核大小。默认值: 3。
        spatio_kernel_dec (int): 解码器卷积核大小。默认值: 3。
        drop (float): Dropout 概率。默认值: 0.0。
        drop_path (float): Stochastic Depth 概率。默认值: 0.0。
        d_state (int): Mamba SSM 的状态维度。默认值: 16。
        d_conv (int): Mamba SSM 的局部卷积核大小。默认值: 4。
        expand (int): Mamba SSM 的扩展系数。默认值: 2。
        
        # --- 训练与优化参数 (Training & Optimization) ---
        lr (float): 学习率。默认值: 1e-3。
        opt (str): 优化器名称 (如 "adamw")。默认值: "adamw"。
        weight_decay (float): 权重衰减系数。默认值: 1e-2。
        filter_bias_and_bn (bool): 是否对 Bias 和 BN 层参数禁用权重衰减。默认值: False。
        sched (str): 学习率调度器名称 (如 "cosine")。默认值: "cosine"。
        min_lr (float): 最小学习率。默认值: 1e-5。
        warmup_lr (float): 预热起始学习率。默认值: 1e-5。
        warmup_epoch (int): 学习率预热的 Epoch 数。默认值: 5。
        decay_epoch (int): 学习率衰减的 Epoch 数。默认值: 30。
        decay_rate (float): 学习率衰减率。默认值: 0.1。
        use_curriculum_learning (bool): 是否启用课程学习策略调整 Loss 权重。默认值: True。
        max_epochs (int): 最大训练 Epoch 数。默认值: 100。
        
        # --- 损失权重参数 (Loss Weights) ---
        loss_weight_l1 (float): L1 Loss 初始权重。默认值: 1.0。
        loss_weight_ssim (float): SSIM Loss 初始权重。默认值: 0.5。
        loss_weight_csi (float): CSI (Critical Success Index) Loss 初始权重。默认值: 1.0。
        loss_weight_spectral (float): 频谱损失权重。默认值: 0.1。
        loss_weight_evo (float): 演变一致性损失权重。默认值: 0.5。
        
        # --- 其他参数 (Misc) ---
        resize_shape (Optional[Tuple[int, int]]): 训练/推理时的目标空间尺寸 (H, W)。
                                                 如果为 None，则使用输入数据的尺寸。
    """
    def __init__(
        self,
        # 1. Input/Output Shapes
        in_shape: Tuple[int, int, int, int] = (10, 54, 256, 256),
        out_seq_length: int = 20,
        out_channels: int = 1,
        
        # 2. Model Architecture
        hid_S: int = 128,
        hid_T: int = 512,
        N_S: int = 4,
        N_T: int = 12,
        mlp_ratio: float = 4.0,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        drop: float = 0.0,
        drop_path: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        
        # 3. Training & Optimization
        lr: float = 1e-3,
        opt: str = "adamw",
        weight_decay: float = 1e-2,
        filter_bias_and_bn: bool = False,
        sched: str = "cosine",
        min_lr: float = 1e-5,
        warmup_lr: float = 1e-5,
        warmup_epoch: int = 5,
        decay_epoch: int = 30,
        decay_rate: float = 0.1,
        use_curriculum_learning: bool = True,
        max_epochs: int = 100,
        
        # 4. Loss Weights
        loss_weight_l1: float = 1.0,
        loss_weight_ssim: float = 0.5,
        loss_weight_csi: float = 1.0,
        loss_weight_spectral: float = 0.1,
        loss_weight_evo: float = 0.5,
        
        # 5. Misc
        resize_shape: Optional[Tuple[int, int]] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 构建 MetMamba 模型
        self.model = self._build_model(self.hparams)
        
        # 2. 初始化混合 Loss 模块
        # 包含 L1, SSIM, CSI (分类), Spectral, Evolution 等多种损失的加权组合
        self.criterion = HybridLoss(
            l1_weight=self.hparams.loss_weight_l1,
            ssim_weight=self.hparams.loss_weight_ssim,
            csi_weight=self.hparams.loss_weight_csi,
            spectral_weight=self.hparams.loss_weight_spectral,
            evo_weight=self.hparams.loss_weight_evo
        )
        
        # 3. 初始化验证评分器
        # MetScore 用于计算气象领域的常用指标 (CSI, POD, FAR 等)
        # nn.Module 的子类会自动被 Lightning 管理设备 (Device)
        self.val_scorer = MetScore()
        
        if resize_shape is None and in_shape is not None:
             # 默认使用输入数据的 H, W
             self.resize_shape = (in_shape[2], in_shape[3])
        else:
             self.resize_shape = resize_shape
             
        self.use_curriculum = self.hparams.use_curriculum_learning

    def _build_model(self, config):
        """根据配置构建 MetMamba 模型实例。"""
        def get_cfg(key, default=None):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        return MetMamba(
            in_shape=get_cfg('in_shape'),
            hid_S=get_cfg('hid_S', 128),
            hid_T=get_cfg('hid_T', 512),
            N_S=get_cfg('N_S', 4),
            N_T=get_cfg('N_T', 12),
            mlp_ratio=get_cfg('mlp_ratio', 4.0),
            drop=get_cfg('drop', 0.0),
            drop_path=get_cfg('drop_path', 0.0),
            spatio_kernel_enc=get_cfg('spatio_kernel_enc', 3),
            spatio_kernel_dec=get_cfg('spatio_kernel_dec', 3),
            out_channels=get_cfg('out_channels', 1),
            out_seq_length=get_cfg('out_seq_length', 20),
            d_state=get_cfg('d_state', 16),
            d_conv=get_cfg('d_conv', 4),
            expand=get_cfg('expand', 2),
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """配置优化器和学习率调度器。

        使用 `metai.model.core.get_optim_scheduler` 工具函数创建。
        """
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, self.hparams.get('max_epochs', 100), self.model
        )
        return cast(OptimizerLRScheduler, {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch" if by_epoch else "step"},
        })
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        """自定义学习率调度器步进逻辑。

        兼容 Timm Scheduler (需要 epoch 参数) 和 PyTorch Native Scheduler。
        """
        if metric is None:
            # Case 1: 基于 Epoch/Step 的调度器 (Cosine, StepLR 等)
            try:
                # 优先尝试 Timm 风格 (必须显式传入 epoch)
                scheduler.step(epoch=self.current_epoch)
            except TypeError:
                scheduler.step()
        else:
            # Case 2: 基于指标的调度器 (ReduceLROnPlateau)
            scheduler.step(metric)
    
    def on_train_epoch_start(self):
        """课程学习 (Curriculum Learning): 动态调整 Loss 权重。

        策略：
        1. 预热期 (Warmup): 仅使用强 L1 Loss 引导模型快速收敛到正确的数值范围，忽略结构细节。
        2. 课程学习期 (Curriculum): 逐渐引入 SSIM (结构相似性) 和 CSI (降水分类指标) 等高级 Loss。
           随着训练进行，L1 权重降低，CSI 权重呈指数级增长，强制模型关注强降水事件的准确性。
        """
        if not self.use_curriculum: return
        
        max_epochs = self.hparams.get('max_epochs', 100)
        # [设定] 预热期长度，建议至少 5-10 个 Epoch
        warmup_epochs = 10  
        
        # === 阶段 1: 预热期 (Warmup Phase) ===
        if self.current_epoch < warmup_epochs:
            weights = {
                'l1': 10.0,  # 强 L1 约束
                'ssim': 0.0, # 暂时关闭
                'csi': 0.0,  # 暂时关闭 (避免阈值梯度干扰)
                'spec': 0.0, # 暂时关闭
                'evo': 0.0   # 暂时关闭
            }
            
        # === 阶段 2: 课程学习期 (Curriculum Phase) ===
        else:
            # 计算当前阶段的进度 (0.0 -> 1.0)
            effective_epoch = self.current_epoch - warmup_epochs
            effective_max = max_epochs - warmup_epochs
            # 防止除以 0
            progress = effective_epoch / effective_max if effective_max > 0 else 1.0
            progress = max(0.0, min(1.0, progress)) # 截断到 [0, 1]
            
            weights = {
                # L1: 从 10.0 快速下降到 1.0 (后期不再过分关注像素平滑)
                'l1': max(10.0 - (9.0 * (progress ** 0.5)), 1.0),
                
                # SSIM: 从 0.0 线性增加到 0.5 (关注结构)
                'ssim': 0.5 * progress,
                
                # CSI: 从 0.0 指数增长到 5.0 (决胜指标，后期权重极大，关注强回波)
                'csi': 5.0 * (progress ** 2),
                
                # Spec: 从 0.0 增加到 0.1 (保持频谱分布)
                'spec': 0.1 * progress,
                
                # Evo: 从 0.0 增加到 0.5 (保持时序演变一致性)
                'evo': 0.5 * progress
            }
        
        # 更新 Loss 模块的内部权重
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights.update(weights)
        
        # 记录日志，监控权重变化
        for k, v in weights.items():
            self.log(f"train/w_{k}", v, on_epoch=True, sync_dist=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入数据 [B, T_in, C, H, W]。

        Returns:
            torch.Tensor: 模型预测输出 [B, T_out, C, H, W]。
        """
        return self.model(x)
    
    def _interp(self, tensor, mode='max_pool'):
        """辅助函数：调整 Tensor 的空间分辨率。

        对于降水数据，通常首选 'max_pool' 进行下采样，以保留局部的强降水极值信息。

        Args:
            tensor (torch.Tensor): 输入张量 [B, T, C, H, W] 或其他 5D 张量。
            mode (str): 插值模式。可选 'max_pool', 'nearest', 'bilinear' 等。

        Returns:
            torch.Tensor: 调整尺寸后的张量 [B, T, C, target_H, target_W]。
        """
        if self.resize_shape is None: return tensor
        B, T, C, H, W = tensor.shape
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return tensor

        # 将 Batch 和 Time 维度合并，以使用 2D 插值/池化函数
        flat = tensor.view(B * T, C, H, W)
        if mode == 'max_pool':
            out = F.adaptive_max_pool2d(flat, (target_H, target_W))
        else:
            out = F.interpolate(flat, size=(target_H, target_W), mode=mode)
        return out.view(B, T, C, target_H, target_W)
    
    def training_step(self, batch, batch_idx):
        """训练步。

        Args:
            batch: Tuple 包含 (_, x, y, _, mask)。
                   x: [B, T_in, C, H, W]
                   y: [B, T_out, C, H, W]
                   mask: [B, T_out, 1, H, W] (有效区域掩码)

        Returns:
            torch.Tensor: 当前 Batch 的 Loss。
        """
        _, x, y, _, mask = batch

        # 对输入、标签和 Mask 进行可能的尺寸调整 (例如下采样训练以加速)
        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        mask = self._interp(mask.float(), 'nearest')

        y_pred = self(x)
        
        # 计算混合 Loss
        loss, _ = self.criterion(y_pred, y, mask=mask)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)


        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步。

        Args:
            batch: Tuple 包含 (_, x, y, _, mask)。
        """
        _, x, y, _, mask = batch

        # 1. 数据对齐与插值
        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        
        # 2. Mask 处理：确保转为布尔型，用于过滤无效区域 (如雷达扫描边界外)
        mask = self._interp(mask.float(), 'nearest')
        mask_bool = mask > 0.5
        
        # 3. 推理
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        # 截断到 [0, 1] 范围
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        # 4. Loss 计算 (应用 Mask)
        loss, _ = self.criterion(logits_pred, y, mask=mask_bool)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 5. 气象指标评分 (CSI, HSS 等)
        # MetScore.forward(pred, target, mask) 会处理反归一化和分级阈值
        scores_dict = self.val_scorer(y_pred_clamped, y, mask=mask_bool)
        
        self.log('val_score', scores_dict['total_score'], on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 6. 计算全局 MAE (Masked)
        if mask_bool is not None:
             val_mae = (torch.abs(y_pred_clamped - y) * mask_bool).sum() / (mask_bool.sum() + 1e-8)
        else:
             val_mae = F.l1_loss(y_pred_clamped, y)
             
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """测试步。

        除了记录 Loss 和 Score，还会返回预测结果供后续可视化或保存。
        """
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')

        mask = self._interp(mask.float(), 'nearest')
        mask_bool = mask > 0.5

        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        with torch.no_grad():
            loss, loss_dict = self.criterion(logits_pred, y, mask=mask_bool)
            scores = self.val_scorer(y_pred_clamped, y, mask=mask_bool)
            
        if self._trainer:
            self.log('test_loss', loss, on_epoch=True)
            self.log('test_score', scores['total_score'], on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),       # [T_in, C, H, W]
            'preds': y_pred_clamped[0].cpu().float().numpy(), # [T_out, C, H, W]
            'trues': y[0].cpu().float().numpy()         # [T_out, C, H, W]
        }
    
    def infer_step(self, batch, batch_idx):
        """纯推理步 (无标签)。

        Args:
            batch: Tuple 包含 (metadata, x, input_mask)。

        Returns:
            torch.Tensor: 预测结果 [B, T_out, C, H, W]，值域 [0, 1]。
        """
        metadata, x, mask = batch 
        x = self._interp(x, mode='max_pool')
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning 的 predict 接口封装。"""
        return self.infer_step(batch, batch_idx)