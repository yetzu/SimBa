# metai/model/mamba/module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from mamba_ssm import Mamba

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    相比 LayerNorm，RMSNorm 不减去均值，更适合处理稀疏数据（如雷达回波中的大量零值）。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., C] or [..., C, H, W] depending on usage, usually [..., C]
        # Calculate RMS
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class TokenSpaceMLP(nn.Module):
    """Token Space MLP with RMSNorm support"""
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptiveFusionGate(nn.Module):
    """自适应门控融合模块"""
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
        
    def forward(self, h_feat, v_feat):
        combined_context = (h_feat + v_feat) / 2.0
        context = combined_context.mean(dim=(1, 2)) 
        gate = self.gate_net(context).unsqueeze(1).unsqueeze(1)
        return gate * h_feat + (1 - gate) * v_feat

class SpatialMambaBlock(nn.Module):
    """
    Spatial Mamba Block (SS2D) - 升级版
    使用 RMSNorm 替代 LayerNorm。
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

    def forward(self, x):
        # x: [B*T, C, H, W]
        B, C, H, W = x.shape
        L = H * W
        
        # [B, C, H, W] -> [B, H, W, C]
        x_hw = x.permute(0, 2, 3, 1).contiguous()
        x_token = x_hw.view(B, -1, C)
        
        # Norm
        x_norm = self.norm1(x_token)
        x_hw_norm = x_norm.view(B, H, W, C)
        
        # SS2D Scanning
        # 1. Horizontal
        x_h_flat = x_hw_norm.view(B, -1, C) 
        out_h_fwd = self.mamba_h(x_h_flat)
        out_h_bwd = self.mamba_h(x_h_flat.flip([1])).flip([1])
        out_h_combined = (out_h_fwd + out_h_bwd).view(B, H, W, C)
        
        # 2. Vertical
        x_v_flat = x_hw_norm.transpose(1, 2).contiguous().view(B, -1, C)
        out_v_fwd = self.mamba_v(x_v_flat)
        out_v_bwd = self.mamba_v(x_v_flat.flip([1])).flip([1])
        out_v_combined = (out_v_fwd + out_v_bwd).view(B, W, H, C).transpose(1, 2)
        
        # Fusion
        mamba_out_hw = self.fusion_gate(out_h_combined, out_v_combined)
        mamba_out = mamba_out_hw.view(B, L, C)
        
        # Residual 1
        x_token = x_token + self.drop_path(mamba_out)
        
        # MLP
        x_token_norm = self.norm2(x_token)
        mlp_out = self.mlp(x_token_norm)
        
        # Residual 2
        x_token = x_token + self.drop_path(mlp_out)
        
        # Restore -> [B, C, H, W]
        x_out = x_token.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out

class TemporalMambaBlock(nn.Module):
    """
    Temporal Mamba Block (Explicit Temporal Scanning)
    显式对时间轴 T 进行扫描。
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
        # 时间轴只需要一个方向或者双向，这里使用标准单向，增强因果性，或者双向增强理解
        # 考虑到是 MidNet（中间层），我们可以用双向来理解整个输入序列的上下文
        self.mamba_t = Mamba(**mamba_cfg) 
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TokenSpaceMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Input x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # Prepare for Temporal Scan: Treat (B, H, W) as batch, T as sequence
        # [B, T, C, H, W] -> [B, H, W, T, C] -> [B*H*W, T, C]
        x_t = x.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, T, C)
        
        # Norm
        x_norm = self.norm1(x_t)
        
        # Temporal Mamba Scan
        # 可以是双向: Fwd + Bwd
        out_fwd = self.mamba_t(x_norm)
        out_bwd = self.mamba_t(x_norm.flip([1])).flip([1])
        mamba_out = out_fwd + out_bwd
        
        # Residual 1
        x_t = x_t + self.drop_path(mamba_out)
        
        # MLP
        x_t_norm = self.norm2(x_t)
        mlp_out = self.mlp(x_t_norm)
        
        # Residual 2
        x_t = x_t + self.drop_path(mlp_out)
        
        # Restore: [B*H*W, T, C] -> [B, T, C, H, W]
        x_out = x_t.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return x_out