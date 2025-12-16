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
    MetMamba Lightning Trainer
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
        
        # 2. 初始化混合 Loss
        self.criterion = HybridLoss(
            l1_weight=self.hparams.loss_weight_l1,
            ssim_weight=self.hparams.loss_weight_ssim,
            csi_weight=self.hparams.loss_weight_csi,
            spectral_weight=self.hparams.loss_weight_spectral,
            evo_weight=self.hparams.loss_weight_evo
        )
        
        # 3. 初始化验证评分器 (nn.Module 会自动管理 Device)
        self.val_scorer = MetScore()
        
        if resize_shape is None and in_shape is not None:
             self.resize_shape = (in_shape[2], in_shape[3])
        else:
             self.resize_shape = resize_shape
             
        self.use_curriculum = self.hparams.use_curriculum_learning

    def _build_model(self, config):
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
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, self.hparams.get('max_epochs', 100), self.model
        )
        return cast(OptimizerLRScheduler, {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch" if by_epoch else "step"},
        })
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        """
        兼容 Timm Scheduler 和 PyTorch Native Scheduler 的 Step 逻辑
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
        """Curriculum Learning: Dynamic Loss Weights (With Warmup)"""
        if not self.use_curriculum: return
        
        max_epochs = self.hparams.get('max_epochs', 100)
        # [新增] 设定预热期，建议至少 5-10 个 Epoch
        warmup_epochs = 10  
        
        # === 阶段 1: 预热期 (Warmup Phase) ===
        # 目标：仅使用强 L1 Loss 引导模型快速收敛到正确的数值范围
        if self.current_epoch < warmup_epochs:
            weights = {
                'l1': 10.0,  # 强 L1 约束
                'ssim': 0.0, # 暂时关闭
                'csi': 0.0,  # 暂时关闭 (避免阈值梯度干扰)
                'spec': 0.0, # 暂时关闭
                'evo': 0.0   # 暂时关闭
            }
            
        # === 阶段 2: 课程学习期 (Curriculum Phase) ===
        # 目标：模型成型后，逐渐引入结构(SSIM)和指标(CSI)优化
        else:
            # 重新计算进度 (0.0 -> 1.0)，基于剩余的 Epoch
            effective_epoch = self.current_epoch - warmup_epochs
            effective_max = max_epochs - warmup_epochs
            # 防止除以 0
            progress = effective_epoch / effective_max if effective_max > 0 else 1.0
            progress = max(0.0, min(1.0, progress)) # 截断到 [0, 1]
            
            weights = {
                # L1: 从 10.0 快速下降到 1.0 (后期不再过分关注像素平滑)
                'l1': max(10.0 - (9.0 * (progress ** 0.5)), 1.0),
                
                # SSIM: 从 0.0 线性增加到 0.5
                'ssim': 0.5 * progress,
                
                # CSI: 从 0.0 指数增长到 5.0 (决胜指标，后期权重极大)
                'csi': 5.0 * (progress ** 2),
                
                # Spec: 从 0.0 增加到 0.1
                'spec': 0.1 * progress,
                
                # Evo: 从 0.0 增加到 0.5
                'evo': 0.5 * progress
            }
        
        # 更新 Loss 模块的权重
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights.update(weights)
        
        # 记录日志
        for k, v in weights.items():
            self.log(f"train/w_{k}", v, on_epoch=True, sync_dist=True)

    def forward(self, x):
        return self.model(x)
    
    def _interp(self, tensor, mode='max_pool'):
        if self.resize_shape is None: return tensor
        B, T, C, H, W = tensor.shape
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return tensor

        flat = tensor.view(B * T, C, H, W)
        if mode == 'max_pool':
            out = F.adaptive_max_pool2d(flat, (target_H, target_W))
        else:
            out = F.interpolate(flat, size=(target_H, target_W), mode=mode)
        return out.view(B, T, C, target_H, target_W)
    
    def training_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        
        mask = self._interp(mask.float(), 'nearest')

        y_pred = self(x)
        loss, _ = self.criterion(y_pred, y, mask=mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        # 1. 数据对齐与插值
        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        
        # 2. Mask 处理：确保转为布尔型 (MetScore 内部会再次检查，但传入 float 0/1 也没问题)
        mask = self._interp(mask.float(), 'nearest')
        mask_bool = mask > 0.5
        
        # 3. 推理
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        # 4. Loss 计算 (Mask 已经生效)
        loss, _ = self.criterion(logits_pred, y, mask=mask_bool)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # MetScore.forward(pred, target, mask) 
        # 它会自动处理反归一化、Mask过滤、时效权重和分级权重
        scores_dict = self.val_scorer(y_pred_clamped, y, mask=mask_bool)
        
        self.log('val_score', scores_dict['total_score'], on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 6. 计算全局 MAE (Masked)
        # MAE: 使用 Bool Mask 计算有效区域
        if mask_bool is not None:
             val_mae = (torch.abs(y_pred_clamped - y) * mask_bool).sum() / (mask_bool.sum() + 1e-8)
        else:
             val_mae = F.l1_loss(y_pred_clamped, y)
             
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
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
            # 同样可以在测试时记录 Score
            scores = self.val_scorer(y_pred_clamped, y, mask=mask_bool)
            
        if self._trainer:
            self.log('test_loss', loss, on_epoch=True)
            self.log('test_score', scores['total_score'], on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }
    
    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        x = self._interp(x, mode='max_pool')
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)