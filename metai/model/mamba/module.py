# metai/model/mamba/module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from mamba_ssm import Mamba

class RMSNorm(nn.Module):
    """均方根层归一化 (Root Mean Square Layer Normalization).

    相比 LayerNorm，RMSNorm 省略了减去均值的操作，计算效率更高，且更适合处理具有稀疏特性的数据
    （例如气象雷达回波中的大量零值区域）。

    Args:
        dim (int): 输入的特征维度 (Channel dimension).
        eps (float, optional): 防止除零的极小值. 默认值: 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状通常为 [..., C].

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同 [..., C].
        """
        # 计算特征维度的均方根
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        # 乘以逆标准差进行缩放
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class TokenSpaceMLP(nn.Module):
    """支持 RMSNorm 的 Token Space 多层感知机 (MLP).

    标准的 MLP 结构：FC -> Act -> Dropout -> FC -> Dropout.

    Args:
        in_features (int): 输入特征维度.
        hidden_features (int, optional): 隐藏层特征维度. 默认为 None (等于 in_features).
        out_features (int, optional): 输出特征维度. 默认为 None (等于 in_features).
        act_layer (nn.Module, optional): 激活函数层. 默认值: nn.GELU.
        drop (float, optional): Dropout 概率. 默认值: 0.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 [..., in_features].

        Returns:
            torch.Tensor: 输出特征，形状为 [..., out_features].
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptiveFusionGate(nn.Module):
    """自适应门控融合模块 (Adaptive Fusion Gate).

    用于融合来自不同扫描方向（如水平和垂直）的特征。通过全局上下文信息生成门控权重，
    实现特征的加权求和。

    Args:
        dim (int): 输入特征通道数.
        reduction (int, optional): 中间层缩减比例，用于控制参数量. 默认值: 4.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        gate_dim = max(dim // reduction, 8)
        self.gate_net = nn.Sequential(
            nn.Linear(dim, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, dim),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    def forward(self, h_feat: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_feat (torch.Tensor): 水平扫描特征，形状 [B, H, W, C].
            v_feat (torch.Tensor): 垂直扫描特征，形状 [B, H, W, C].

        Returns:
            torch.Tensor: 融合后的特征，形状 [B, H, W, C].
        """
        # 简单平均作为上下文基础
        combined_context = (h_feat + v_feat) / 2.0
        # 全局平均池化: [B, H, W, C] -> [B, C]
        context = combined_context.mean(dim=(1, 2)) 
        # 生成门控系数: [B, C] -> [B, 1, 1, C]
        gate = self.gate_net(context).unsqueeze(1).unsqueeze(1)
        # 加权融合
        return gate * h_feat + (1 - gate) * v_feat

class SpatialMambaBlock(nn.Module):
    """空间 Mamba 模块 (Spatial Mamba Block / SS2D).

    采用 SS2D (Spatial Scanning 2D) 机制，将 2D 图像平展为序列进行 Mamba 处理。
    包含水平 (Horizontal) 和垂直 (Vertical) 两个方向的扫描，并使用 RMSNorm 替代 LayerNorm。

    Args:
        dim (int): 输入特征维度.
        mlp_ratio (float, optional): MLP 隐藏层扩展比例. 默认值: 4.0.
        drop (float, optional): Dropout 概率. 默认值: 0.0.
        drop_path (float, optional): Stochastic Depth 概率. 默认值: 0.0.
        act_layer (nn.Module, optional): 激活函数. 默认值: nn.GELU.
        d_state (int, optional): SSM 状态维度 (Latent State Dim). 默认值: 16.
        d_conv (int, optional): SSM 局部卷积核大小. 默认值: 4.
        expand (int, optional): Mamba 内部维度扩展系数. 默认值: 2.
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        mamba_cfg = dict(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_h = Mamba(**mamba_cfg)
        self.mamba_v = Mamba(**mamba_cfg)
        
        self.fusion_gate = AdaptiveFusionGate(dim=dim, reduction=4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TokenSpaceMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征，形状 [B*T, C, H, W]. 
                              注意：此处 Batch 维度通常混合了时间维度 (Batch * Time)。

        Returns:
            torch.Tensor: 输出特征，形状 [B*T, C, H, W].
        """
        B, C, H, W = x.shape
        L = H * W
        
        # 维度变换: [B, C, H, W] -> [B, H, W, C]
        x_hw = x.permute(0, 2, 3, 1).contiguous()
        # 展平为 Token 序列: [B, L, C]
        x_token = x_hw.view(B, -1, C)
        
        # 第一层 Norm
        x_norm = self.norm1(x_token)
        x_hw_norm = x_norm.view(B, H, W, C)
        
        # --- SS2D 扫描机制 ---
        
        # 1. 水平扫描 (Horizontal Scan)
        # 将每一行视为一个序列: [B, H*W, C] (行优先)
        x_h_flat = x_hw_norm.view(B, -1, C) 
        # 前向扫描
        out_h_fwd = self.mamba_h(x_h_flat)
        # 后向扫描 (翻转序列 -> 处理 -> 翻转回来)
        out_h_bwd = self.mamba_h(x_h_flat.flip([1])).flip([1])
        # 双向结果叠加
        out_h_combined = (out_h_fwd + out_h_bwd).view(B, H, W, C)
        
        # 2. 垂直扫描 (Vertical Scan)
        # 转置 H 和 W，使每一列变为连续序列: [B, W, H, C] -> [B, W*H, C]
        x_v_flat = x_hw_norm.transpose(1, 2).contiguous().view(B, -1, C)
        out_v_fwd = self.mamba_v(x_v_flat)
        out_v_bwd = self.mamba_v(x_v_flat.flip([1])).flip([1])
        # 恢复维度: [B, W, H, C] -> [B, H, W, C]
        out_v_combined = (out_v_fwd + out_v_bwd).view(B, W, H, C).transpose(1, 2)
        
        # --- 特征融合 ---
        
        # 自适应门控融合水平和垂直特征
        mamba_out_hw = self.fusion_gate(out_h_combined, out_v_combined)
        mamba_out = mamba_out_hw.view(B, L, C)
        
        # 残差连接 1
        x_token = x_token + self.drop_path(mamba_out)
        
        # --- MLP ---
        
        x_token_norm = self.norm2(x_token)
        mlp_out = self.mlp(x_token_norm)
        
        # 残差连接 2
        x_token = x_token + self.drop_path(mlp_out)
        
        # 恢复输出形状 -> [B, C, H, W]
        x_out = x_token.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out

class TemporalMambaBlock(nn.Module):
    """时间 Mamba 模块 (Temporal Mamba Block).

    显式对时间轴 T 进行建模。将空间维度 (H, W) 视为 Batch 的一部分，
    对每个空间位置的时间序列进行 Mamba 扫描。

    Args:
        dim (int): 输入特征维度.
        mlp_ratio (float, optional): MLP 扩展比例. 默认值: 4.0.
        drop (float, optional): Dropout 概率. 默认值: 0.0.
        drop_path (float, optional): Stochastic Depth 概率. 默认值: 0.0.
        d_state (int, optional): SSM 状态维度. 默认值: 16.
        d_conv (int, optional): SSM 局部卷积核大小. 默认值: 4.
        expand (int, optional): Mamba 扩展系数. 默认值: 2.
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        mamba_cfg = dict(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # 时间轴建模，使用双向 Mamba 以充分利用整个序列的上下文信息（适用于中间层 MidNet）
        self.mamba_t = Mamba(**mamba_cfg) 
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TokenSpaceMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征，形状 [B, T, C, H, W].

        Returns:
            torch.Tensor: 输出特征，形状 [B, T, C, H, W].
        """
        B, T, C, H, W = x.shape
        
        # --- 时间轴扫描准备 ---
        # 我们希望对 T 维度进行序列建模。
        # 将 (B, H, W) 视为独立的样本，T 视为序列长度。
        # [B, T, C, H, W] -> [B, H, W, T, C] -> [B*H*W, T, C]
        x_t = x.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, T, C)
        
        # Norm
        x_norm = self.norm1(x_t)
        
        # --- Temporal Mamba Scan ---
        # 双向扫描 (Bidirectional): Forward + Backward
        out_fwd = self.mamba_t(x_norm)
        out_bwd = self.mamba_t(x_norm.flip([1])).flip([1])
        mamba_out = out_fwd + out_bwd
        
        # 残差连接 1
        x_t = x_t + self.drop_path(mamba_out)
        
        # --- MLP ---
        x_t_norm = self.norm2(x_t)
        mlp_out = self.mlp(x_t_norm)
        
        # 残差连接 2
        x_t = x_t + self.drop_path(mlp_out)
        
        # 恢复维度: [B*H*W, T, C] -> [B, H, W, T, C] -> [B, T, C, H, W]
        x_out = x_t.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return x_out