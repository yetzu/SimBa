# metai/model/mamba/model.py
import torch
from torch import nn
from timm.layers.weight_init import trunc_normal_

from .module import MambaSubBlock

class BasicConv2d(nn.Module):
    """
    基础卷积块：Conv2d + GroupNorm + SiLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, upsampling=False, act_norm=False, act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """
    卷积跳跃连接块 (Stride Convolution)
    用于 Encoder 下采样和 Decoder 上采样
    """
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, 
                 upsampling=False, act_norm=True, act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        return self.conv(x)


def sampling_generator(N, reverse=False):
    """生成下采样/上采样配置列表"""
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """
    独立实现的 Encoder
    负责将输入特征图进行空间下采样
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """
    独立实现的 Decoder
    负责将特征图上采样回原始分辨率
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        # 融合 Encoder 第一层的特征 (Skip Connection)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MetaBlock(nn.Module):
    """
    SimVP MetaBlock (mamba-only).

    Important for checkpoint compatibility:
    - Keep attribute names: `block` and optional `reduction`
    - Keep forward behavior identical to simvp's MetaBlock when model_type == 'mamba'
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        input_resolution=None,
        model_type=None,
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        layer_i=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        model_type_l = model_type.lower() if model_type is not None else "gsta"
        if model_type_l != "mamba":
            raise ValueError(f"metai.model.mamba.MetaBlock only supports model_type='mamba', got {model_type!r}")

        self.block = MambaSubBlock(
            in_channels,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
        )

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """
    The hidden Translator of MetaFormer for SimVP (mamba-only).

    Copied behavior from `metai/model/simvp/simvp_model.py::MidMetaNet` to preserve checkpoint keys.
    """

    def __init__(
        self,
        channel_in,
        channel_hid,
        N2,
        input_resolution=None,
        model_type=None,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
        channel_out=None,
    ):
        super().__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2

        if channel_out is None:
            channel_out = channel_in
        self.channel_out = channel_out

        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # 1. Input layer
        enc_layers = [
            MetaBlock(
                channel_in,
                channel_hid,
                input_resolution,
                model_type,
                mlp_ratio,
                drop,
                drop_path=dpr[0],
                layer_i=0,
            )
        ]

        # 2. Middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(
                MetaBlock(
                    channel_hid,
                    channel_hid,
                    input_resolution,
                    model_type,
                    mlp_ratio,
                    drop,
                    drop_path=dpr[i],
                    layer_i=i,
                )
            )

        # 3. Output layer
        enc_layers.append(
            MetaBlock(
                channel_hid,
                channel_out,
                input_resolution,
                model_type,
                mlp_ratio,
                drop,
                drop_path=drop_path,
                layer_i=N2 - 1,
            )
        )

        # Keep name `enc` for checkpoint compatibility
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        # x: [B, T_in, C, H, W] -> [B, T_in*C, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        # z: [B, T_out*C, H, W] -> [B, T_out, C, H, W]
        T_out = self.channel_out // C
        y = z.reshape(B, T_out, C, H, W)
        return y


class SimVP_Model(nn.Module):
    r"""
    SimVP Model (mamba mode) migrated into `metai/model/mamba`.

    This keeps the original SimVP(mamba) structure and module naming so that checkpoints
    trained with `metai/model/simvp` (model_type='mamba') can be loaded and run here.
    """

    def __init__(
        self,
        in_shape,
        hid_S=16,
        hid_T=256,
        N_S=4,
        N_T=4,
        model_type="mamba",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=True,
        out_channels=None,
        aft_seq_length=None,
        **kwargs,
    ):
        super().__init__()

        # T: input frames, C: input channels
        T, C, H, W = in_shape
        self.T_out = aft_seq_length if aft_seq_length is not None else T

        if out_channels is None:
            out_channels = C
        self.out_channels = out_channels

        # Downsampled resolution (same logic as SimVP)
        H_ds, W_ds = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))
        act_inplace = False

        # Encoder / Decoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type_l = "gsta" if model_type is None else str(model_type).lower()
        if model_type_l != "mamba":
            raise ValueError(f"metai.model.mamba.SimVP_Model only supports model_type='mamba', got {model_type!r}")

        channel_in = T * hid_S
        channel_out = self.T_out * hid_S
        self.hid = MidMetaNet(
            channel_in,
            hid_T,
            N_T,
            input_resolution=(H_ds, W_ds),
            model_type=model_type_l,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            channel_out=channel_out,
        )

        # Readout
        self.readout = nn.Conv2d(hid_S, out_channels, kernel_size=1)
        if self.readout.bias is not None:
            nn.init.constant_(self.readout.bias, -5.0)

    def forward(self, x_raw, **kwargs):
        # x_raw: [B, T_in, C_in, H, W]
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        # 1. Encoder
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        # 2. Translator (MidMetaNet)
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)  # [B, T_out, C_, H_, W_]
        hid = hid.reshape(B * self.T_out, C_, H_, W_)

        # 3. Skip alignment (same as SimVP)
        _, C_skip, H_skip, W_skip = skip.shape
        skip = skip.view(B, T, C_skip, H_skip, W_skip)
        skip_last = skip[:, -1:, ...]
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1).reshape(B * self.T_out, C_skip, H_skip, W_skip)

        # 4. Decoder + Readout
        Y = self.dec(hid, skip_out)
        Y = self.readout(Y)
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        return Y

class MambaMidNet(nn.Module):
    """
    专为 Mamba 设计的中间层网络
    堆叠多个 MambaSubBlock 来提取时空特征
    """
    def __init__(self, channel_in, channel_hid, N2, mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MambaMidNet, self).__init__()
        assert N2 >= 2
        
        # 降维/升维层 (如果输入通道与隐藏通道不一致)
        self.channel_in = channel_in
        self.channel_hid = channel_hid
        
        # 输入投影
        self.proj_in = nn.Conv2d(channel_in, channel_hid, 1, 1, 0) if channel_in != channel_hid else nn.Identity()
        
        # 堆叠 Mamba Block
        # 使用线性衰减的 drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, N2)]
        
        layers = []
        for i in range(N2):
            layers.append(MambaSubBlock(
                dim=channel_hid, 
                mlp_ratio=mlp_ratio, 
                drop=drop, 
                drop_path=dpr[i]
            ))
        self.layers = nn.Sequential(*layers)
        
        # 输出投影 (如果需要还原通道数，通常 MidNet 输出给 Decoder 前会保持 hidden dim，但在 SimVP 架构中通常不需要变回 T*C)
        # 这里我们保持输出为 channel_hid，由模型主类处理 reshape
        
    def forward(self, x):
        # x: [B, T*C, H, W]
        z = self.proj_in(x)
        z = self.layers(z)
        return z


class SimMamba(nn.Module):
    r"""
    Sim Mamba 完全体模型 (Standalone)
    
    结构:
    1. Encoder: 空间下采样
    2. MambaMidNet: 时空特征学习 (核心)
    3. Decoder: 空间上采样 + 预测
    """

    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=8, 
                 mlp_ratio=4., drop=0.0, drop_path=0.0, 
                 spatio_kernel_enc=3, spatio_kernel_dec=3, 
                 act_inplace=True, out_channels=None, aft_seq_length=None, **kwargs):
        super(SimMamba, self).__init__()
        
        # T: 输入帧数, C: 输入通道数
        T, C, H, W = in_shape  
        
        # 确定输出帧数
        self.T_out = aft_seq_length if aft_seq_length is not None else T
        if out_channels is None:
            out_channels = C
        self.out_channels = out_channels
        
        # 1. Encoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        
        # 2. Translator (Mamba)
        # Encoder 输出的通道数为 hid_S，堆叠 T 帧后为 T * hid_S
        mid_channel_in = T * hid_S
        # 这是一个超参数，决定中间层的宽度，通常设为 256 或 512
        mid_channel_hid = hid_T 
        
        self.hid = MambaMidNet(
            channel_in=mid_channel_in, 
            channel_hid=mid_channel_hid, 
            N2=N_T, 
            mlp_ratio=mlp_ratio, 
            drop=drop, 
            drop_path=drop_path
        )
        
        # 3. Decoder
        # MambaMidNet 输出通道为 mid_channel_hid (hid_T)
        # 但 Decoder 期望的输入需要在时间维度展开，即 T_out * hid_S
        # 因此需要一个投影层将 hid_T 映射到 T_out * hid_S
        self.proj_out = nn.Conv2d(mid_channel_hid, self.T_out * hid_S, 1, 1, 0)
        
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        # 4. Readout
        self.readout = nn.Conv2d(hid_S, out_channels, kernel_size=1)
        if self.readout.bias is not None:
            nn.init.constant_(self.readout.bias, -5.0) # 初始化偏置以优化收敛

    def forward(self, x_raw, **kwargs):
        # x_raw: [B, T, C, H, W]
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        # 1. Encoder
        # embed: [B*T, hid_S, H', W']
        # skip:  [B*T, hid_S, H, W]
        embed, skip = self.enc(x)
        _, _, H_, W_ = embed.shape 

        # 2. Translator (Mamba)
        # [B*T, hid_S, H', W'] -> [B, T*hid_S, H', W']
        z = embed.view(B, T * -1, H_, W_) 
        hid = self.hid(z) # -> [B, hid_T, H', W']
        
        # 投影到输出序列长度: [B, T_out*hid_S, H', W']
        hid = self.proj_out(hid)
        
        # 3. Decoder
        # Reshape 为 [B*T_out, hid_S, H', W'] 以喂入 Decoder
        dec_in = hid.view(B * self.T_out, -1, H_, W_)

        # 处理 Skip Connection (对齐 T_out)
        # skip: [B*T, hid_S, H, W] -> [B, T, hid_S, H, W]
        skip = skip.view(B, T, -1, H, W)
        # 取最后一帧的 skip 特征: [B, 1, hid_S, H, W]
        skip_last = skip[:, -1:, ...] 
        # 复制 T_out 次: [B, T_out, hid_S, H, W] -> [B*T_out, hid_S, H, W]
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1).reshape(B * self.T_out, -1, H, W)
        
        # 解码
        Y = self.dec(dec_in, skip_out)
        
        # 4. Readout
        Y = self.readout(Y)
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        
        return Y