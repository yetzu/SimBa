# metai/model/mamba/model.py

import torch
from torch import nn
from timm.layers.weight_init import trunc_normal_
from .module import MambaSubBlock

class BasicConv2d(nn.Module):
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
        
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
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

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        return self.readout(Y)

class MambaMidNet(nn.Module):
    """纯粹的 Mamba 中间特征提取网络"""
    def __init__(self, channel_in, channel_hid, N2, mlp_ratio=4., drop=0., drop_path=0., d_state=16, d_conv=4, expand=2):
        super().__init__()
        # 通道投影: T*hid_S -> hid_T
        self.proj_in = nn.Conv2d(channel_in, channel_hid, 1, 1, 0) if channel_in != channel_hid else nn.Identity()
        
        # Mamba 堆叠层
        dpr = [x.item() for x in torch.linspace(0, drop_path, N2)]
        self.layers = nn.Sequential(*[
            MambaSubBlock(dim=channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=dpr[i], d_state=d_state, d_conv=d_conv, expand=expand)
            for i in range(N2)
        ])
    
    def forward(self, x):
        z = self.proj_in(x)
        z = self.layers(z)
        return z

class MetMamba(nn.Module):
    """
    MetMamba (原 SimMamba)
    纯净架构: Encoder -> MambaMidNet -> Decoder
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
        
        # 2. Temporal Translator (Mamba)
        # 输入维度: T帧 * 空间维度hid_S
        mid_channel_in = T * hid_S
        self.hid = MambaMidNet(mid_channel_in, hid_T, N_T, mlp_ratio, drop, drop_path, d_state, d_conv, expand)
        
        # 输出投影: hid_T -> T_out * hid_S
        self.proj_out = nn.Conv2d(hid_T, self.T_out * hid_S, 1, 1, 0)
        
        # 3. Spatial Decoder
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec)
        self.readout = nn.Conv2d(hid_S, out_channels, 1)
        
        # Bias 初始化优化
        if self.readout.bias is not None:
            nn.init.constant_(self.readout.bias, -5.0)

    def forward(self, x_raw):
        # x_raw: [B, T, C, H, W]
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        # 1. Encode
        # embed: [B*T, hid_S, H_, W_]
        # skip:  [B*T, hid_S, H_skip, W_skip]
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape 

        # 2. Translate (Time-Mixing via Mamba)
        # 显式重塑为 5D: [B, T, C_, H_, W_]
        z = embed.view(B, T, C_, H_, W_)
        # 展平为 4D 供 MambaMidNet 处理: [B, T*C_, H_, W_]
        z_flat = z.reshape(B, T * C_, H_, W_)
        
        hid = self.hid(z_flat)       # -> [B, hid_T, H_, W_]
        hid = self.proj_out(hid)     # -> [B, T_out*C_, H_, W_]
        
        # 3. Decode
        # 重塑为 [B*T_out, C_, H_, W_] 供 Decoder 处理
        dec_in = hid.reshape(B * self.T_out, C_, H_, W_)
        
        # Skip Connection Alignment (对齐 SimVP 实现)
        # 获取 skip 自身的空间维度
        _, C_skip, H_skip, W_skip = skip.shape
        
        # 还原 skip 的时间维度: [B, T, C_skip, H_skip, W_skip]
        skip = skip.view(B, T, C_skip, H_skip, W_skip)
        
        # 策略：取 Encoder 最后一帧的 Skip 特征，复制 T_out 次
        skip_last = skip[:, -1:, ...] # [B, 1, C_skip, H_skip, W_skip]
        
        # 扩展到 T_out 长度并展平: [B*T_out, C_skip, H_skip, W_skip]
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1)
        skip_out = skip_out.reshape(B * self.T_out, C_skip, H_skip, W_skip)
        
        Y = self.dec(dec_in, skip_out)
        Y = self.readout(Y)
        
        return Y.reshape(B, self.T_out, -1, H, W)