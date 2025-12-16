# metai/model/mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


class WeightedScoreSoftCSILoss(nn.Module):
    """加权区间 Soft-CSI 损失函数 (Weighted Interval Soft-CSI Loss).

    该损失函数旨在直接优化气象预测中的 CSI (Critical Success Index) 指标。
    由于 CSI 是不可导的（基于 0/1 混淆矩阵），这里使用了 Sigmoid 函数来近似 0/1 阶跃，
    从而实现可微的 Soft-CSI。

    改进说明：
    严格对齐了 `metrices.py` 中的区间评分逻辑：
    - 区间阈值 (low <= pred < high)。对应 Bin-based 规则，更精准地约束降水强度落在正确的区间内。

    Args:
        smooth (float, optional): 平滑因子，防止分母为 0. 默认值: 1.0.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0 
        
        # --- 1. 对齐强度分级及权重 ---
        # 阈值: 0.1, 1.0, 2.0, 5.0, 8.0 (mm)
        thresholds_raw = [0.1, 1.0, 2.0, 5.0, 8.0]
        # 权重: 0.1, 0.1, 0.2, 0.25, 0.35
        weights_raw    = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        # 注册归一化后的下界 (Low Thresholds) [L]
        self.register_buffer('thresholds', torch.tensor(thresholds_raw) / self.MM_MAX)
        
        # 注册归一化后的上界 (High Thresholds) [L]
        # 构造逻辑：[1.0, 2.0, 5.0, 8.0, inf]
        # 使得每个 Bin 对应 [0.1, 1.0), [1.0, 2.0), ...
        highs_raw = thresholds_raw[1:] + [float('inf')]
        self.register_buffer('highs', torch.tensor(highs_raw) / self.MM_MAX)
        
        # 注册分级权重 [L]
        self.register_buffer('intensity_weights', torch.tensor(weights_raw))
        
        # --- 2. 对齐时效及权重 ---
        # 随预测时长变化的权重，通常中间时刻权重较大
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        # [1, 20, 1, 1]
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): 预测值 (归一化后 0~1)，形状 [B, T, H, W]。
            target (torch.Tensor): 真实值 (归一化后 0~1)，形状 [B, T, H, W]。
            mask (torch.Tensor, optional): 有效区域掩码，形状 [B, T, H, W] 或 [B, 1, H, W]。

        Returns:
            torch.Tensor: 加权后的 Soft-CSI Loss (标量)。
        """
        T = pred.shape[1]
        # 截取当前序列长度对应的时间权重并归一化
        current_time_weights = self.time_weights[:, :T, :, :]
        current_time_weights = current_time_weights / current_time_weights.mean()
        
        # Mask 维度对齐
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
            elif mask.dim() == 5:
                mask = mask.squeeze(2)

        total_weighted_loss = 0.0
        total_weight_sum = 0.0

        # 同时遍历下界(low)和上界(high)计算每个 Bin 的 CSI Loss
        for i, (t_low, t_high) in enumerate(zip(self.thresholds, self.highs)):
            w = self.intensity_weights[i]
            
            # --- 1. 计算区间 Soft Probability ---
            # 逻辑：Prob(区间) = Sigmoid(pred - low) * (1 - Sigmoid(pred - high))
            # 含义：预测值必须 "软" 大于下界，且 "软" 小于上界
            
            # (A) 大于下界的概率 (使用 steepness=2000 来模拟阶跃函数)
            score_low = torch.sigmoid((pred - t_low) * 2000)
            
            # (B) 小于上界的概率 (如果是 inf 则概率为 1)
            if torch.isinf(t_high):
                score_in_bin = score_low # 最后一个区间只看下界 [8.0, inf)
            else:
                score_high = torch.sigmoid((pred - t_high) * 2000)
                # "在区间内" = "大于下界" AND "不大于上界"
                score_in_bin = score_low * (1.0 - score_high)

            # --- 2. 计算区间 Target ---
            # 逻辑：Target = (target >= low) & (target < high)
            # 真实值是离散/确定的，直接使用布尔运算
            target_ge_low = (target >= t_low)
            if torch.isinf(t_high):
                target_in_bin = target_ge_low.float()
            else:
                target_lt_high = (target < t_high)
                target_in_bin = (target_ge_low & target_lt_high).float()
            
            # --- 3. 应用 Mask ---
            if mask is not None:
                score_in_bin = score_in_bin * mask
                target_in_bin = target_in_bin * mask
                
            # --- 4. 计算 Soft-CSI ---
            # 在空间维度 (H, W) 上求和，保留 (B, T)
            intersection = (score_in_bin * target_in_bin).sum(dim=(-2, -1))
            total_pred = score_in_bin.sum(dim=(-2, -1))
            total_target = target_in_bin.sum(dim=(-2, -1))
            union = total_pred + total_target - intersection
            
            # CSI = Intersection / Union
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi 
            
            # 应用时间权重并取平均
            weighted_loss_t = (loss_map * current_time_weights.squeeze(-1).squeeze(-1)).mean()
            
            total_weighted_loss += weighted_loss_t * w
            total_weight_sum += w

        return total_weighted_loss / total_weight_sum

class LogSpectralDistanceLoss(nn.Module):
    """对数谱距离损失 (Log Spectral Distance Loss).

    通过计算预测图像和真实图像在频域幅度的对数差，来约束图像的纹理细节。
    这有助于缓解 L1/L2 Loss 导致的图像模糊问题。

    Args:
        epsilon (float, optional): 防止对数计算负无穷的极小值. 默认值: 1e-6.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: 
        """
        Args:
            pred (torch.Tensor): 预测值 [B, T, H, W].
            target (torch.Tensor): 真实值 [B, T, H, W].
            mask (torch.Tensor, optional): 掩码 [B, T, H, W].

        Returns:
            torch.Tensor: 频域损失 (标量).
        """
        # 1. 预处理：应用 Mask
        # 必须在 FFT 前将无效区域置为 0，否则 FFT 会把背景的填充值（可能是随机值或NaN）
        # 转换为全频段噪声，严重干扰 Loss 计算。
        if mask is not None:
            # 扩展 Mask 维度以匹配 [B, T, H, W]
            if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
                mask_bc = mask.expand(-1, pred.shape[1], -1, -1)
            elif mask.dim() == 5:
                mask_bc = mask.squeeze(2)
            else:
                mask_bc = mask
                
            pred = pred * mask_bc
            target = target * mask_bc
        
        # FFT 变换需要 float32
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        
        # 2D FFT 变换 (实数输入 -> 复数频谱)
        # dim=(-2, -1) 表示对 H, W 维度进行变换
        pred_fft = torch.fft.rfft2(pred_fp32, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target_fp32, dim=(-2, -1), norm='ortho')
        
        # 计算幅度谱 (Magnitude Spectrum)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 计算对数距离 (L1 Loss on Log Magnitude)
        loss = F.l1_loss(torch.log(pred_mag + self.epsilon), torch.log(target_mag + self.epsilon))
        
        return loss


class WeightedEvolutionLoss(nn.Module):
    """加权物理感知演变损失 (Weighted Evolution Loss).

    作用：约束气象系统的时序演变连贯性（一阶时间差分约束），并重点关注强回波区的变化。
    防止预测出的降水系统发生不自然的突变。

    Args:
        weight_scale (float, optional): 强回波区域的额外权重系数. 默认值: 5.0.
    """
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): 预测序列 [B, T, H, W].
            target (torch.Tensor): 真实序列 [B, T, H, W].
            mask (torch.Tensor, optional): 掩码 [B, T, H, W].

        Returns:
            torch.Tensor: 演变损失 (标量).
        """
        # 计算时间差分 (dI/dt): 后一帧减前一帧
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # 计算演变误差 (L1 距离)
        diff_error = torch.abs(pred_diff - target_diff)
        
        # 动态加权：如果该位置是强回波，则赋予更高权重
        # 逻辑：强回波的移动和生消是预测难点，也是业务重点。
        # 使用 target[:, 1:] (t+1时刻) 的强度来决定权重。
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        # 应用 Mask
        if mask is not None:
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            
            # 取 T-1 帧的 Mask (代表 t+1 时刻的有效性，因为 diff 使得长度减 1)
            mask_t_plus_1 = mask[:, 1:] 
            
            diff_error = diff_error * mask_t_plus_1 
            weight_map = weight_map * mask_t_plus_1 
            
            count_valid = mask_t_plus_1.sum()
            if count_valid > 0:
                weighted_loss = (diff_error * weight_map).sum() / count_valid
            else:
                # [FIXED] 返回 Tensor 而不是 float，避免 .item() 报错，并保持计算图
                weighted_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        else:
            weighted_loss = (diff_error * weight_map).mean()

        return weighted_loss


class HybridLoss(nn.Module):
    """物理感知混合损失函数 (Physics-aware Hybrid Loss).

    组合了多种 Loss 以兼顾空间精准度、结构纹理、评分指标和时序连贯性：
    1. L1 Loss (with Hard Example Mining): 像素级误差，对强降水区域进行强力加权。
    2. Soft-CSI Loss: 直接优化竞赛评估指标。
    3. Spectral Loss: 保持频域纹理，防止模糊。
    4. Evolution Loss: 保持时序演变的物理合理性。
    5. MS-SSIM Loss: 保持多尺度的结构相似性。

    Args:
        l1_weight (float): L1 损失权重.
        ssim_weight (float): MS-SSIM 损失权重.
        csi_weight (float): Soft-CSI 损失权重.
        spectral_weight (float): 频谱损失权重.
        evo_weight (float): 演变损失权重.
    """
    def __init__(self, 
                 l1_weight=1.0, 
                 ssim_weight=0.5, 
                 csi_weight=1.0, 
                 spectral_weight=0.1, 
                 evo_weight=0.5):
        super().__init__()
        self.weights = {
            'l1': l1_weight,
            'ssim': ssim_weight,
            'csi': csi_weight,
            'spec': spectral_weight,
            'evo': evo_weight
        }
        
        # 必须使用 reduction='none' 才能支持后续的 Pixel-Wise 加权和 Masking
        self.l1 = nn.L1Loss(reduction='none') 
        
        if ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None
            
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            logits (torch.Tensor): 模型输出的原始 Logits [B, T, C, H, W] 或 [B, T, H, W].
            target (torch.Tensor): 归一化后的真实值 [B, T, C, H, W] (范围 0~1).
            mask (torch.Tensor, optional): 有效掩码 [B, T, C, H, W] 或 [B, T, H, W].

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss (torch.Tensor): 加权总损失.
                - loss_dict (dict): 各分项损失值 (float).
        """
        # 1. 维度预处理: 移除 Channel 维度 (如果是 1)
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)
        
        # 将 logits 转为 [0, 1] 概率
        pred = torch.sigmoid(logits)
        
        loss_dict = {}
        total_loss = 0.0
        
        # =====================================================================
        # 2. L1 Loss (Pixel-Wise) - 难例挖掘 (Hard Example Mining)
        # =====================================================================
        # 计算基础 L1 误差
        l1_loss_map = self.l1(pred, target) # [B, T, H, W]
        
        # 动态权重：基于赛题评分规则对强降水区域进行暴力加权
        # 目标：强迫模型关注稀疏但高价值的强降水区域
        # 归一化基准: MM_MAX = 30.0
        
        pixel_weight = torch.ones_like(target)
        
        # Level 1: > 2.0mm (权重 0.1 -> 0.2) -> 设为 x2 关注度
        # 2.0 / 30.0 = 0.0667
        pixel_weight[target > (2.0 / 30.0)] = 2.0
        
        # Level 2: > 5.0mm (权重 0.2 -> 0.25) -> 设为 x5 关注度
        # 5.0 / 30.0 = 0.1667
        pixel_weight[target > (5.0 / 30.0)] = 5.0
        
        # Level 3: > 8.0mm (权重 0.25 -> 0.35, 最高分) -> 设为 x50 关注度 !!!
        # 8.0 / 30.0 = 0.2667
        # 策略：这是决胜点。即使产生一些虚警(FP)，也要保证能抓到强回波(TP)。
        # 对于极少数的极端降水，赋予极高的惩罚权重。
        pixel_weight[target > (8.0 / 30.0)] = 50.0
        
        # 应用动态权重
        l1_loss_map = l1_loss_map * pixel_weight
        
        # 应用有效区域 Mask
        if mask is not None:
            masked_error = l1_loss_map * mask
            count_valid = mask.sum()
            # 避免除以 0
            l1_loss = masked_error.sum() / (count_valid + 1e-8)
        else:
            l1_loss = l1_loss_map.mean()
            
        total_loss += self.weights['l1'] * l1_loss
        
        # 增加类型检查，防止 .item() 报错
        loss_dict['l1'] = l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        # =====================================================================
        
        # 3. Soft-CSI Loss (直接优化评价指标)
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target, mask)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item() if isinstance(csi_loss, torch.Tensor) else csi_loss
            
        # 4. Spectral Loss (频域抗模糊)
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target, mask)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item() if isinstance(spec_loss, torch.Tensor) else spec_loss
            
        # 5. Evolution Loss (时序演变约束)
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target, mask)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item() if isinstance(evo_loss, torch.Tensor) else evo_loss
            
        # 6. MS-SSIM Loss (结构一致性)
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            # SSIM 需要 [B, C, H, W] 格式，这里视 T 为 Batch 的一部分或 Channel
            # 为了计算简便，我们将 (B*T) 视为 Batch
            pred_c = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_c = target.view(-1, 1, target.shape[-2], target.shape[-1])
            
            if mask is not None:
                mask_c = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
                pred_c = pred_c * mask_c
                target_c = target_c * mask_c
            
            ssim_val = self.ms_ssim(pred_c, target_c).mean()
            ssim_loss = 1.0 - ssim_val
            total_loss += self.weights['ssim'] * ssim_loss
            loss_dict['ssim'] = ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else ssim_loss
        
        # 记录加权后的总 Loss
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict