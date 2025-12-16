# metai/model/mamba/model.py

import torch
from torch import nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from .module import SpatialMambaBlock, TemporalMambaBlock, RMSNorm

class BasicConv2d(nn.Module):
    """基础 2D 卷积模块.

    BasicConv2d 集成了 RMSNorm 和 SiLU 激活函数，支持上采样操作。

    Args:
        in_channels (int): 输入通道数.
        out_channels (int): 输出通道数.
        kernel_size (int, optional): 卷积核大小. 默认值: 3.
        stride (int, optional): 步长. 默认值: 1.
        padding (int, optional): 填充. 默认值: 0.
        dilation (int, optional): 膨胀系数. 默认值: 1.
        upsampling (bool, optional): 是否进行 2 倍上采样 (PixelShuffle). 默认值: False.
        act_norm (bool, optional): 是否应用 Norm 和 Activation. 默认值: False.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, upsampling=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            # 使用 PixelShuffle 进行 2x 上采样，卷积输出通道需扩大 4 倍
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size, stride, padding, dilation),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # 替换为 RMSNorm。由于 RMSNorm 通常作用于最后一维 (Channel Last)，
        # 前向传播时需要进行 Permute 操作。
        self.norm = RMSNorm(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量 [B, C_in, H, W].

        Returns:
            torch.Tensor: 输出张量 [B, C_out, H', W'].
        """
        y = self.conv(x)
        if self.act_norm:
            # RMSNorm 期望输入为 [..., C]，因此需要先调整维度顺序
            y = y.permute(0, 2, 3, 1) # [B, H, W, C]
            y = self.norm(y)
            y = y.permute(0, 3, 1, 2) # [B, C, H, W]
            y = self.act(y)
        return y

class ConvSC(nn.Module):
    """卷积层 (用于 Encoder/Decoder 的构建块).

    Args:
        C_in (int): 输入通道数.
        C_out (int): 输出通道数.
        kernel_size (int, optional): 卷积核大小. 默认值: 3.
        downsampling (bool, optional): 是否下采样 (Stride=2). 默认值: False.
        upsampling (bool, optional): 是否上采样 (PixelShuffle). 默认值: False.
        act_norm (bool, optional): 是否应用激活和归一化. 默认值: True.
    """
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        # 自动计算 padding 以保持特征图尺寸 (当 stride=1 时) 或标准下采样
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)
    def forward(self, x): return self.conv(x)

def sampling_generator(N, reverse=False):
    """生成下采样/上采样配置列表.

    例如 N=4, 产生 [False, True, False, True]，表示每隔一层进行一次采样操作。
    """
    samplings = [False, True] * (N // 2)
    return list(reversed(samplings[:N])) if reverse else samplings[:N]

class Encoder(nn.Module):
    """空间编码器 (Spatial Encoder).

    将输入帧编码为潜在特征表示。

    Args:
        C_in (int): 输入图像通道数.
        C_hid (int): 隐藏层通道数.
        N_S (int): 编码器层数.
        spatio_kernel (int): 空间卷积核大小.
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 输入 [B*T, C_in, H, W].

        Returns:
            latent (torch.Tensor): 最终编码特征 [B*T, C_hid, H', W'].
            enc1 (torch.Tensor): 第一层编码特征 [B*T, C_hid, H, W] (用于 Skip Connection).
        """
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class MotionGuidedSkip(nn.Module):
    """光流引导跳跃连接模块 (Motion-Guided Skip Connection).

    核心思想：由于降水云团在不断移动，直接将 Encoder 的过去特征 (Skip Feature) 
    与 Decoder 的当前特征 (Decoder Feature) 进行拼接可能会导致空间错位。
    本模块利用 Decoder 特征（包含未来预测信息）预测光流，将 Encoder 特征 Warp 到当前时刻的位置。

    Args:
        channel_in (int): Skip 特征通道数 (Encoder 输出).
        channel_ref (int): 参考特征通道数 (Decoder 特征).
    """
    def __init__(self, channel_in, channel_ref):
        super().__init__()
        # 光流预测网络
        self.flow_net = nn.Sequential(
            nn.Conv2d(channel_ref + channel_in, channel_in, 3, 1, 1), 
            nn.SiLU(),
            nn.Conv2d(channel_in, 2, 3, 1, 1) # 输出 2 通道 (dx, dy)
        )
        # 初始化光流层为 0，保证训练初始阶段不发生随机偏移，利于收敛
        nn.init.constant_(self.flow_net[-1].weight, 0)
        nn.init.constant_(self.flow_net[-1].bias, 0)

    def warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """应用光流对特征图进行 Warp 操作.

        Args:
            x (torch.Tensor): 待 Warp 的特征图 [B, C, H, W].
            flow (torch.Tensor): 光流场 [B, 2, H, W].

        Returns:
            torch.Tensor: Warp 后的特征图 [B, C, H, W].
        """
        B, C, H, W = x.shape
        # 生成基础网格 (Grid)
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(x.device)
        
        # 应用光流偏移: Grid + Flow
        vgrid = grid + flow
        
        # 归一化网格坐标到 [-1, 1] 区间，以适配 grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1) # [B, H, W, 2]
        # 双线性插值采样
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return output

    def forward(self, skip_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip_feat (torch.Tensor): Encoder 最后一帧的特征 [B, C, H, W].
            dec_feat (torch.Tensor): Decoder 当前层的特征 [B, C, H, W] (蕴含对未来的预测).

        Returns:
            torch.Tensor: 对齐 (Warped) 后的 Skip 特征 [B, C, H, W].
        """
        # 拼接特征并预测光流
        combined_feat = torch.cat([dec_feat, skip_feat], dim=1) 
        flow = self.flow_net(combined_feat)

        # 执行 Warp
        warped_skip = self.warp(skip_feat, flow)
        return warped_skip

class Decoder(nn.Module):
    """空间解码器 (Spatial Decoder).

    将隐藏层特征解码为预测图像，并应用光流引导的 Skip Connection。

    Args:
        C_hid (int): 隐藏层通道数.
        C_out (int): 输出通道数 (通常等于 hidden，最后再做 readout).
        N_S (int): 解码器层数.
        spatio_kernel (int): 空间卷积核大小.
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        
        # 光流引导模块: 用于对齐 Encoder 的特征
        self.motion_skip = MotionGuidedSkip(channel_in=C_hid, channel_ref=C_hid)
    
    def forward(self, hid: torch.Tensor, enc1: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hid (torch.Tensor): 主体输入特征 [B*T_out, C, H, W].
            enc1 (torch.Tensor, optional): 来自 Encoder 的浅层特征 [B*T_out, C, H, W].
                                           (通常是将 Encoder 最后一帧特征复制 T_out 份).

        Returns:
            torch.Tensor: 解码输出 [B*T_out, C_out, H, W].
        """
        # 1. 解码主体 (前 N-1 层)
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
            
        # 2. 光流引导融合 (在最后一层之前)
        # hid 此时包含了未来的位置信息，enc1 是过去的特征。
        # 使用 hid 预测光流来对齐 enc1，使其移动到 hid 指示的位置。
        enc1_aligned = self.motion_skip(enc1, hid)
        
        # 融合并进行最后一次上采样/卷积
        Y = self.dec[-1](hid + enc1_aligned)
        return self.readout(Y)

class STMambaMidNet(nn.Module):
    """ST-Mamba 中间网络 (ST-Mamba MidNet).

    交替堆叠 Spatial Mamba Block 和 Temporal Mamba Block，实现时空特征的深度交互。

    Args:
        channel_hid (int): 隐藏层通道数.
        N2 (int): 堆叠层数 (Spatial + Temporal 的总数).
        mlp_ratio (float): MLP 扩展比例.
        drop (float): Dropout 概率.
        drop_path (float): Stochastic Depth 概率.
        d_state (int): SSM 状态维度.
        d_conv (int): SSM 卷积核大小.
        expand (int): SSM 扩展系数.
    """
    def __init__(self, channel_hid, N2, mlp_ratio=4., drop=0., drop_path=0., d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, N2)]
        
        self.layers = nn.ModuleList([])
        for i in range(N2):
            # 交替策略：偶数层为 Spatial，奇数层为 Temporal
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征 [B, T, C, H, W].

        Returns:
            torch.Tensor: 输出特征 [B, T, C, H, W].
        """
        B, T, C, H, W = x.shape
        
        for layer in self.layers:
            if isinstance(layer, SpatialMambaBlock):
                # Spatial Block 期望输入合并 Time 和 Batch: [B*T, C, H, W]
                x_in = x.view(B*T, C, H, W)
                x_out = layer(x_in)
                x = x_out.view(B, T, C, H, W)
            elif isinstance(layer, TemporalMambaBlock):
                # Temporal Block 直接处理 [B, T, C, H, W]
                x = layer(x)
                
        return x

class MetMamba(nn.Module):
    """MetMamba 主模型.

    架构特点:
    1. ST-Mamba Temporal Modeling: 使用交替的空间和时间 Mamba 块进行时空建模。
    2. Flow-Guided Skip Connection: 使用光流引导机制解决编解码器特征的空间错位问题。
    3. RMSNorm for Sparsity: 使用 RMSNorm 优化稀疏数据的归一化。

    Args:
        in_shape (tuple): 输入张量形状 (T, C, H, W).
        hid_S (int, optional): 空间编码器隐藏层通道数. 默认值: 64.
        hid_T (int, optional): 时空转换模块 (MidNet) 通道数. 默认值: 256.
        N_S (int, optional): Encoder/Decoder 层数. 默认值: 4.
        N_T (int, optional): ST-Mamba MidNet 层数. 默认值: 8.
        mlp_ratio (float, optional): MLP 扩展比例. 默认值: 4.0.
        spatio_kernel_enc (int, optional): Encoder 卷积核大小. 默认值: 3.
        spatio_kernel_dec (int, optional): Decoder 卷积核大小. 默认值: 3.
        out_channels (int, optional): 输出通道数. 默认值: None (等于输入通道).
        out_seq_length (int, optional): 预测序列长度. 默认值: None (等于输入长度).
        d_state (int, optional): SSM 状态维度. 默认值: 16.
        d_conv (int, optional): SSM 卷积核大小. 默认值: 4.
        expand (int, optional): SSM 扩展系数. 默认值: 2.
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
        
        # 2. ST-Translator (时空转换模块)
        # 投影层: 将空间特征映射到高维通道以进行时空混合
        # 输入维度: [B, T, hid_S, H_, W_] -> 映射到 hid_T 通道
        self.proj_in = nn.Conv2d(hid_S, hid_T, 1)
        
        # 核心时空 Mamba 网络
        self.hid = STMambaMidNet(hid_T, N_T, mlp_ratio, drop, drop_path, d_state, d_conv, expand)
        
        # 输出投影层: 
        # 此时 hid 输出为 [B, T, hid_T, H, W]
        # 目标: 映射到 [B, T_out, hid_S, H, W]
        # 策略: Flatten T*hid_T -> 1x1 Conv -> T_out*hid_S (时序预测的通道混合)
        self.proj_out = nn.Conv2d(T * hid_T, self.T_out * hid_S, 1, 1, 0)
        
        # 3. Spatial Decoder (with Motion Skip)
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec)
        self.readout = nn.Conv2d(hid_S, out_channels, 1)
        
        # 初始化输出层 Bias，有助于训练稳定性
        if self.readout.bias is not None:
            nn.init.constant_(self.readout.bias, -5.0)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw (torch.Tensor): 输入序列 [B, T, C, H, W].

        Returns:
            torch.Tensor: 预测序列 [B, T_out, C_out, H, W].
        """
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        # 1. Encode (对每一帧独立编码)
        # embed: [B*T, hid_S, H_, W_] (H_, W_ 为下采样后的尺寸)
        # skip: [B*T, hid_S, H, W] (第一层特征，用于 Skip Connection)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape 
        
        # Reshape back to [B, T, C_, H_, W_]
        z = embed.view(B, T, C_, H_, W_)
        
        # 2. Translate (显式时空混合)
        # 投影到 Mamba 隐藏层维度: [B*T, C_, H_, W_] -> [B*T, hid_T, H_, W_]
        z_in = self.proj_in(z.view(B*T, C_, H_, W_)).view(B, T, -1, H_, W_)
        
        # 经过 ST-Mamba MidNet 处理: [B, T, hid_T, H_, W_]
        hid = self.hid(z_in)       
        
        # 3. Decode Preparation (特征重组以适配输出序列长度)
        # Flatten Temporal and Channel 维度: [B, T*hid_T, H_, W_]
        hid_flat = hid.reshape(B, T * hid.shape[2], H_, W_)
        
        # 投影并预测未来序列长度: [B, T_out*hid_S, H_, W_]
        dec_in = self.proj_out(hid_flat)
        dec_in = dec_in.reshape(B * self.T_out, -1, H_, W_)
        
        # --- Skip Connection 处理 ---
        # 目标：将 Encoder 最后一帧的特征作为 Skip Feature，并复制 T_out 份
        
        # 获取 skip 自身的空间维度
        _, C_skip, H_skip, W_skip = skip.shape
        skip = skip.view(B, T, C_skip, H_skip, W_skip)
        
        # 取最后一帧: [B, 1, C_skip, H_skip, W_skip]
        skip_last = skip[:, -1:, ...] 
        
        # 扩展到 T_out 长度: [B, T_out, C_skip, H_skip, W_skip]
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1)
        # 合并 Batch 和 Time: [B*T_out, C_skip, H_skip, W_skip]
        skip_out = skip_out.reshape(B * self.T_out, C_skip, H_skip, W_skip)
        
        # 4. Decode (包含光流引导对齐)
        # Decoder 内部会自动应用 MotionGuidedSkip，利用 dec_in 预测的光流来对齐 skip_out
        Y = self.dec(dec_in, skip_out)
        Y = self.readout(Y)
        
        # 最终输出 reshape: [B, T_out, C_out, H, W]
        return Y.reshape(B, self.T_out, -1, H, W)