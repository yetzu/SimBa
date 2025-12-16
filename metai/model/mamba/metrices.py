# metai/model/mamba/metrices.py

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Any

class MetScore(nn.Module):
    """竞赛评分计算器模块 (MetScore).

    该模块实现了象竞赛的综合评分公式。
    它支持所有指标（TS, MAE, Correlation）均在物理空间 (mm) 进行计算。

    评分公式逻辑：
    Score = Sum_t( w_t * Correlation_t * Sum_k( w_k * TS_tk * Exp_MAE_tk ) )
    其中：
    - w_t: 时间步权重 (随预报时效变化).
    - w_k: 降水等级权重 (随降水量级变化).
    - Correlation_t: 第 t 时刻的空间相关系数.
    - TS_tk: 第 t 时刻、第 k 个阈值等级的 Threat Score.
    - Exp_MAE_tk: 基于 MAE 的惩罚项 (MAE 越小，该项越接近 1).

    Args:
        data_max (float): 数据反归一化的最大值因子 (通常对应数据集的最大降水量). 默认值: 30.0.
    """
    
    def __init__(self, data_max: float = 30.0):
        super().__init__()
        self.data_max = data_max
        
        # --- 注册常量参数 (Buffer) ---
        # 1. 时效权重 (对应 6min - 120min，共 20 个时间步)
        # 权重设计通常反映了对不同预报时效的重视程度 (如中间时刻权重较高)
        time_weights = torch.tensor([
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ])
        self.register_buffer('time_weights_default', time_weights)

        # 2. 分级阈值 (mm)
        # 用于计算 TS 评分和 MAE 的降水等级划分
        thresholds = torch.tensor([0.1, 1.0, 2.0, 5.0, 8.0])
        self.register_buffer('thresholds', thresholds)

        # 3. 分级区间上界 (用于 Interval TS 计算)
        # 构造区间 [0.1, 1.0), [1.0, 2.0), ..., [8.0, inf)
        inf_tensor = torch.tensor([float('inf')])
        highs = torch.cat([thresholds[1:], inf_tensor])
        self.register_buffer('highs', highs)

        # 4. 分级权重
        # 越大的降水量级通常权重越高，以强调对强降水预报的能力
        level_weights = torch.tensor([0.1, 0.1, 0.2, 0.25, 0.35])
        self.register_buffer('level_weights', level_weights)

    def forward(self, 
                pred_norm: torch.Tensor, 
                target_norm: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算各项评分指标.

        Args:
            pred_norm (torch.Tensor): 归一化后的预测值 (0~1).
                                      支持形状: [B, T, H, W] 或 [B, T, C, H, W].
            target_norm (torch.Tensor): 归一化后的真实值 (0~1).
            mask (torch.Tensor, optional): 有效区域掩码. 
                                           1 (或 True) 表示有效，0 表示无效。

        Returns:
            Dict[str, torch.Tensor]: 包含各项指标的字典 (total_score, ts_time, mae_time 等).
        """
        with torch.no_grad():
            return self._compute(pred_norm, target_norm, mask)

    def _compute(self, pred_norm, target_norm, mask):
        """核心计算逻辑"""
        # 1. 反归一化：还原为物理量 (mm)
        pred = pred_norm * self.data_max
        target = target_norm * self.data_max
        
        # 2. 维度适配处理 (统一为 [B, T, H, W])
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2); target = target.squeeze(2)
        if mask is not None and mask.dim() == 5 and mask.shape[2] == 1:
            mask = mask.squeeze(2)
        if pred.dim() == 3: # [B, H, W] -> [B, 1, H, W]
            pred = pred.unsqueeze(1); target = target.unsqueeze(1)
            if mask is not None and mask.dim() == 3: mask = mask.unsqueeze(1)

        B, T, H, W = pred.shape
        device = pred.device
        
        # 3. 处理 Mask
        if mask is None:
            valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)
        else:
            if mask.shape != pred.shape:
                if mask.dim() == 4 and mask.shape[1] == 1:
                     mask = mask.expand(-1, T, -1, -1)
            valid_mask = mask > 0.5

        scores_k_list, r_k_list, ts_matrix_list, mae_matrix_list = [], [], [], []

        # 4. 逐时间步计算指标
        for k in range(T):
            # 展平当前帧的所有像素: [B*H*W]
            p_k, t_k, m_k = pred[:, k, ...].flatten(), target[:, k, ...].flatten(), valid_mask[:, k, ...].flatten()
            
            # === Correlation (相关系数) ===
            # 过滤掉双零点 (预测和真实值均小于最小阈值)，避免背景噪声干扰相关性计算
            min_thresh = self.thresholds[0]
            is_double_zero = (p_k < min_thresh) & (t_k < min_thresh)
            mask_corr = m_k & (~is_double_zero)
            
            if mask_corr.sum() > 0:
                p_c, t_c = p_k[mask_corr], t_k[mask_corr]
                p_mean, t_mean = p_c.mean(), t_c.mean()
                num = ((p_c - p_mean) * (t_c - t_mean)).sum()
                den = torch.sqrt(((p_c - p_mean)**2).sum() * ((t_c - t_mean)**2).sum())
                R_k = torch.clamp(num / (den + 1e-6), -1.0, 1.0)
            else:
                R_k = torch.tensor(0.0, device=device)
            r_k_list.append(R_k)

            # === TS (Threat Score) & MAE (分级绝对误差) ===
            # 仅考虑掩码有效区域
            p_v, t_v = p_k[m_k], t_k[m_k]
            L = self.thresholds.shape[0]
            
            if p_v.numel() == 0:
                ts_vec, mae_vec = torch.zeros(L, device=device), torch.zeros(L, device=device)
            else:
                # 扩展维度以进行广播比较: [1, N] vs [L, 1]
                p_ex, t_ex = p_v.unsqueeze(0), t_v.unsqueeze(0)
                low_b, high_b = self.thresholds.unsqueeze(1), self.highs.unsqueeze(1)
                
                # 判断像素点落在哪个区间 [low, high)
                is_p_in = (p_ex >= low_b) & (p_ex < high_b)
                is_t_in = (t_ex >= low_b) & (t_ex < high_b)
                
                # 计算列联表 (Contingency Table) 元素
                # Hits: 预测和真实都在该区间
                hits = (is_p_in & is_t_in).float().sum(dim=1)
                # Misses: 真实在该区间，预测不在
                misses = ((~is_p_in) & is_t_in).float().sum(dim=1)
                # False Alarms: 预测在该区间，真实不在
                fas = (is_p_in & (~is_t_in)).float().sum(dim=1)
                
                # TS = Hits / (Hits + Misses + FAs)
                ts_vec = hits / (hits + misses + fas + 1e-8)
                
                # 计算分级 MAE: 仅计算真实值落在该区间的样本的 MAE
                mae_list = []
                for i in range(L):
                    mask_level = is_t_in[i, :] # 该等级的真实样本 Mask
                    if mask_level.sum() > 0:
                        mae_val = (torch.abs(p_v - t_v) * mask_level).sum() / mask_level.sum()
                    else:
                        mae_val = torch.tensor(0.0, device=device)
                    mae_list.append(mae_val)
                mae_vec = torch.stack(mae_list)

            ts_matrix_list.append(ts_vec)
            mae_matrix_list.append(mae_vec)
            
            # === Score (单帧综合评分) ===
            # 1. 相关性项: sqrt(exp(R - 1)) -> R=1时为1，R<1时衰减
            term_corr = torch.sqrt(torch.exp(R_k - 1))
            # 2. MAE 项: sqrt(exp(-MAE / 100)) -> MAE=0时为1，MAE增大时衰减
            term_mae = torch.sqrt(torch.exp(-mae_vec / 100.0))
            # 3. 加权求和: 综合 TS、MAE 和 等级权重
            sum_level_metrics = (self.level_weights * ts_vec * term_mae).sum()
            
            scores_k_list.append(term_corr * sum_level_metrics)

        # 5. 结果聚合
        scores_time = torch.stack(scores_k_list)      # [T]
        ts_time_matrix = torch.stack(ts_matrix_list)  # [T, L]
        mae_time_matrix = torch.stack(mae_matrix_list)# [T, L]
        
        # 6. 时间加权总分
        if T == len(self.time_weights_default): 
            w_time = self.time_weights_default
        elif T < len(self.time_weights_default): 
            # 如果预测步长小于默认权重长度，归一化截断的权重
            w_time = self.time_weights_default[:T] / self.time_weights_default[:T].sum()
        else: 
            w_time = torch.ones(T, device=device) / T
            
        return {
            'total_score': (scores_time * w_time).sum(), # 最终标量分数
            'score_time': scores_time,                   # 每一帧的分数 [T]
            'r_time': torch.stack(r_k_list),             # 每一帧的相关系数 [T]
            'ts_time': ts_time_matrix,                   # [T, L] TS 矩阵
            'mae_time': mae_time_matrix,                 # [T, L] MAE 矩阵
            'ts_levels': ts_time_matrix.mean(dim=0),     # [L] 各等级平均 TS
            'mae_levels': mae_time_matrix.mean(dim=0)    # [L] 各等级平均 MAE
        }

class MetricTracker:
    """指标累加器 (Metric Tracker).
    
    用于在 Validation/Test 过程中累积每个 Batch 的指标值，
    最后计算整个 Epoch 的平均值。
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count = 0
        self.metrics_sum = {}
        
    def update(self, metrics: Dict[str, torch.Tensor]):
        """更新累积器.

        Args:
            metrics (Dict[str, torch.Tensor]): 当前 Batch 的指标字典.
        """
        self.count += 1
        for k, v in metrics.items():
            # 移动到 CPU 以节省 GPU 显存
            val = v.detach().cpu()
            if k not in self.metrics_sum: 
                self.metrics_sum[k] = torch.zeros_like(val)
            self.metrics_sum[k] += val
            
    def compute(self) -> Dict[str, Any]:
        """计算平均值.

        Returns:
            Dict[str, Any]: 平均后的指标字典.
        """
        if self.count == 0: return {}
        return {k: v / self.count for k, v in self.metrics_sum.items()}

class MetMetricCollection(nn.Module):
    """指标集合封装类 (Metric Collection).
    
    功能：
    1. 组合 MetScore (单次计算) 和 MetricTracker (累积统计)。
    2. 处理指标名称前缀 (如 'train_', 'val_')。
    3. 适配 LightningModule 的调用方式，方便集成。

    Args:
        prefix (str): 指标名称前缀 (例如 "val_").
    """
    def __init__(self, prefix: str = ""):
        super().__init__()
        self.prefix = prefix
        # MetScore 是 nn.Module，包含 buffer，Lightning 会自动处理设备移动
        self.scorer = MetScore()
        # Tracker 是普通类，用于 CPU 累积
        self.tracker = MetricTracker()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算当前 Batch 的指标并更新 Tracker。
        
        Args:
            pred (torch.Tensor): 预测值.
            target (torch.Tensor): 真实值.
            mask (torch.Tensor, optional): 掩码.

        Returns:
            Dict[str, torch.Tensor]: 带前缀的当前 Batch 指标 (用于 step log).
        """
        # 1. 计算当前 Batch 指标 (GPU)
        scores = self.scorer(pred, target, mask)
        
        # 2. 更新累积器 (CPU)
        self.tracker.update(scores)
        
        # 3. 返回带前缀的结果
        return {f"{self.prefix}{k}": v for k, v in scores.items()}

    def compute(self) -> Dict[str, Any]:
        """计算累积平均值并返回带前缀的字典.
        
        通常在 on_validation_epoch_end 中调用。
        """
        metrics = self.tracker.compute()
        return {f"{self.prefix}{k}": v for k, v in metrics.items()}

    def reset(self):
        """重置累积器."""
        self.tracker.reset()