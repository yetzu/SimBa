# metai/model/mamba/trainer.py

import torch
import torch.nn.functional as F
import lightning as l
from typing import Any, Dict, cast, Tuple, Optional
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from metai.model.core import get_optim_scheduler
from .model import MetMamba
from .loss import HybridLoss

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
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
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
    
    # =======================================================
    # 新增：重写 lr_scheduler_step 以支持 Timm 调度器
    # =======================================================
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        """
        Custom LR Scheduler Step for TIMM Schedulers
        TIMM 调度器通常需要显式传入 epoch 参数
        """
        scheduler.step(epoch=self.current_epoch)
    
    def on_train_epoch_start(self):
        """Curriculum Learning: Dynamic Loss Weights"""
        if not self.use_curriculum: return
        
        max_epochs = self.hparams.get('max_epochs', 100)
        progress = self.current_epoch / max_epochs
        
        # 动态权重策略
        weights = {
            'l1': max(10.0 - (9.0 * (progress ** 0.5)), 1.0),
            'ssim': 1.0 - 0.5 * progress,
            'csi': 0.5 + 4.5 * (progress ** 2),
            'spec': 0.1 * progress,
            'evo': 0.5 * progress
        }
        
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights.update(weights)
        
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
        mask = mask.bool()
        mask = self._interp(mask.float(), 'nearest')

        pred = self(x)
        loss, loss_dict = self.criterion(pred, y, mask=mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        mask = mask.bool()
        mask = self._interp(mask.float(), 'nearest')
        
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        loss, _ = self.criterion(logits_pred, y, mask=mask)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        MM_MAX = 30.0
        pred_mm = y_pred_clamped * MM_MAX
        target_mm = y * MM_MAX

        thresholds = [0.1, 1.0, 2.0, 5.0, 8.0]
        level_weights = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        time_weights_list = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ]
        
        T_out = pred_mm.shape[1]
        if T_out == 20:
            time_weights = torch.tensor(time_weights_list, device=self.device)
        else:
            time_weights = torch.ones(T_out, device=self.device) / T_out

        total_score = 0.0
        total_level_weight = sum(level_weights)

        for t_val, w_level in zip(thresholds, level_weights):
            if pred_mm.dim() == 5 and pred_mm.shape[2] == 1:
                 p_mm = pred_mm.squeeze(2)
                 t_mm = target_mm.squeeze(2)
            else:
                 p_mm = pred_mm
                 t_mm = target_mm

            hits_tensor = (p_mm >= t_val) & (t_mm >= t_val)
            misses_tensor = (p_mm < t_val) & (t_mm >= t_val)
            false_alarms_tensor = (p_mm >= t_val) & (t_mm < t_val)
            
            if p_mm.dim() == 4:
                sum_dims = (0, 2, 3)
            else: 
                sum_dims = (0, 2, 3, 4)

            hits = hits_tensor.float().sum(dim=sum_dims)
            misses = misses_tensor.float().sum(dim=sum_dims)
            false_alarms = false_alarms_tensor.float().sum(dim=sum_dims)
            
            ts_t = hits / (hits + misses + false_alarms + 1e-6)
            ts_weighted_time = (ts_t * time_weights).sum()
            total_score += ts_weighted_time * w_level

        val_score = total_score / total_level_weight

        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)
        val_mae = F.l1_loss(y_pred_clamped, y)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        mask = mask.bool()
        mask = self._interp(mask.float(), 'nearest')

        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        with torch.no_grad():
            loss, loss_dict = self.criterion(logits_pred, y, mask=mask)
            
        self.log('test_loss', loss, on_epoch=True)
        
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