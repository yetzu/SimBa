# metai/model/mamba/model.py

import torch
from torch import nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from .module import SpatialMambaBlock, TemporalMambaBlock, RMSNorm

class BasicConv2d(nn.Module):
    """
    改进版 BasicConv2d: 使用 RMSNorm 替换 GroupNorm/LayerNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, upsampling=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size, stride, padding, dilation),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # 替换为 RMSNorm (注意 RMSNorm 通常作用于 Channel Last，这里用 1x1 Conv 模拟或 Permute)
        # 为了兼容性，这里我们实现一个支持 2D 图像的 RMSNorm 包装
        self.norm = RMSNorm(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            # RMSNorm expects [..., C], so permute
            y = y.permute(0, 2, 3, 1) # [B, H, W, C]
            y = self.norm(y)
            y = y.permute(0, 3, 1, 2) # [B, C, H, W]
            y = self.act(y)
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)
    def forward(self, x): return self.conv(x)

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    return list(reversed(samplings[:N])) if reverse else samplings[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )
    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class MotionGuidedSkip(nn.Module):
    """
    光流引导跳跃连接模块
    利用 Decoder 的未来特征预测光流，将 Encoder 的过去特征 Warp 到当前位置。
    """
    def __init__(self, channel_in, channel_ref):
        super().__init__()
        # channel_in: Skip 特征通道数
        # channel_ref: Decoder 特征通道数
        self.flow_net = nn.Sequential(
            nn.Conv2d(channel_ref, channel_in, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channel_in, 2, 3, 1, 1) # 输出 (dx, dy)
        )
        # 初始化光流层为0，保证初始状态下不发生偏移
        nn.init.constant_(self.flow_net[-1].weight, 0)
        nn.init.constant_(self.flow_net[-1].bias, 0)

    def warp(self, x, flow):
        # x: [B, C, H, W]
        # flow: [B, 2, H, W]
        B, C, H, W = x.shape
        # 生成 Grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(x.device)
        
        # Apply Flow
        vgrid = grid + flow
        
        # Normalize to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1) # [B, H, W, 2]
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return output

    def forward(self, skip_feat, dec_feat):
        """
        skip_feat: Encoder 最后一帧特征 [B, C, H, W]
        dec_feat: Decoder 当前帧特征 [B, C, H, W] (蕴含未来位置信息)
        """
        # flow = self.flow_net(dec_feat)
        combined_feat = torch.cat([dec_feat, skip_feat], dim=1) 
        flow = self.flow_net(combined_feat)
        
        warped_skip = self.warp(skip_feat, flow)
        return warped_skip

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        
        # 新增: 光流引导模块
        self.motion_skip = MotionGuidedSkip(channel_in=C_hid, channel_ref=C_hid)
    
    def forward(self, hid, enc1=None):
        # hid: [B*T_out, C, H, W]
        # enc1: [B*T_out, C, H, W] (Expanded last frame skip)
        
        # 1. 解码主体
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
            
        # 2. 在最后一层融合前，应用光流 Warp
        # hid 此时包含了未来的位置信息，enc1 是过去的特征
        # 使用 hid 预测光流来对齐 enc1
        enc1_aligned = self.motion_skip(enc1, hid)
        
        Y = self.dec[-1](hid + enc1_aligned)
        return self.readout(Y)

class STMambaMidNet(nn.Module):
    """
    ST-Mamba MidNet (Space-Time Alternating)
    交替堆叠 Spatial Mamba 和 Temporal Mamba
    """
    def __init__(self, channel_hid, N2, mlp_ratio=4., drop=0., drop_path=0., d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, N2)]
        
        self.layers = nn.ModuleList([])
        for i in range(N2):
            # 交替：偶数层 Spatial, 奇数层 Temporal
            if i % 2 == 0:
                self.layers.append(
                    SpatialMambaBlock(dim=channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=dpr[i], 
                                      d_state=d_state, d_conv=d_conv, expand=expand)
                )
            else:
                self.layers.append(
                    TemporalMambaBlock(dim=channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=dpr[i], 
                                       d_state=d_state, d_conv=d_conv, expand=expand)
                )
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 循环处理
        for layer in self.layers:
            if isinstance(layer, SpatialMambaBlock):
                # Spatial Block 期望输入 [B*T, C, H, W]
                x_in = x.view(B*T, C, H, W)
                x_out = layer(x_in)
                x = x_out.view(B, T, C, H, W)
            elif isinstance(layer, TemporalMambaBlock):
                # Temporal Block 期望输入 [B, T, C, H, W]
                x = layer(x)
                
        return x

class MetMamba(nn.Module):
    """
    MetMamba++ (Enhanced Version)
    1. ST-Mamba Temporal Modeling
    2. Flow-Guided Skip Connection
    3. RMSNorm for Sparsity
    """
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=8, 
                 mlp_ratio=4., drop=0.0, drop_path=0.0, 
                 spatio_kernel_enc=3, spatio_kernel_dec=3, 
                 out_channels=None, out_seq_length=None, 
                 d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        T, C, H, W = in_shape
        self.T_out = out_seq_length if out_seq_length is not None else T
        out_channels = out_channels if out_channels is not None else C
        
        # 1. Spatial Encoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # 2. ST-Translator (Explicit Time Modeling)
        # 输入维度: [B, T, hid_S, H_, W_] -> 映射到 hid_T 通道
        self.proj_in = nn.Conv2d(hid_S, hid_T, 1)
        self.hid = STMambaMidNet(hid_T, N_T, mlp_ratio, drop, drop_path, d_state, d_conv, expand)
        
        # 输出投影: 
        # 此时 hid 输出为 [B, T, hid_T, H, W]
        # 我们需要将其映射到 [B, T_out, hid_S, H, W]
        # 策略：Flatten T*hid_T -> 1x1 Conv -> T_out*hid_S
        self.proj_out = nn.Conv2d(T * hid_T, self.T_out * hid_S, 1, 1, 0)
        
        # 3. Spatial Decoder (with Motion Skip)
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec)
        self.readout = nn.Conv2d(hid_S, out_channels, 1)
        
        if self.readout.bias is not None:
            nn.init.constant_(self.readout.bias, -5.0)

    def forward(self, x_raw):
        # x_raw: [B, T, C, H, W]
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        # 1. Encode
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape 
        
        # Reshape back to [B, T, C_, H_, W_]
        z = embed.view(B, T, C_, H_, W_)
        
        # Project to high-dim hidden channel for Mamba
        # Apply to each frame: [B*T, C_, H_, W_] -> [B*T, hid_T, H_, W_]
        z_in = self.proj_in(z.view(B*T, C_, H_, W_)).view(B, T, -1, H_, W_)
        
        # 2. Translate (Explicit Space-Time Mixing)
        # hid: [B, T, hid_T, H_, W_]
        hid = self.hid(z_in)       
        
        # 3. Decode Preparation
        # Flatten Temporal and Channel: [B, T*hid_T, H_, W_]
        hid_flat = hid.reshape(B, T * hid.shape[2], H_, W_)
        
        # Project to output sequence length: [B, T_out*hid_S, H_, W_]
        dec_in = self.proj_out(hid_flat)
        dec_in = dec_in.reshape(B * self.T_out, -1, H_, W_)
        
        # Skip Connection Handling
        # 获取 skip 自身的空间维度
        _, C_skip, H_skip, W_skip = skip.shape
        skip = skip.view(B, T, C_skip, H_skip, W_skip)
        
        # 取最后一帧 [B, 1, C_skip, H_skip, W_skip]
        skip_last = skip[:, -1:, ...] 
        
        # 扩展到 T_out 长度: [B*T_out, C_skip, H_skip, W_skip]
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1)
        skip_out = skip_out.reshape(B * self.T_out, C_skip, H_skip, W_skip)
        
        # Decoder 内部会自动应用 MotionGuidedSkip 对齐 skip_out
        Y = self.dec(dec_in, skip_out)
        Y = self.readout(Y)
        
        return Y.reshape(B, self.T_out, -1, H, W)