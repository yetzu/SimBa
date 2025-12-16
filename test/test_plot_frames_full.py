import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

# ==========================================
# 环境配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath('.'))

try:
    from metai.dataset.met_dataloader_scwds import ScwdsDataModule
    # 尝试导入真实常量
    from metai.utils.met_var import MetLabel, MetRadar, MetNwp, MetGis
    HAS_MET_VAR = True
except ImportError:
    print("[Warning] 无法导入 metai.utils.met_var，使用脚本内置定义。")
    HAS_MET_VAR = False
    # Fallback Definitions
    class MetLabel: RA = "RA"; _min=0; _max=300
    class MetRadar: 
        CR="CR"; CAP20="CAP20"; CAP30="CAP30"; CAP40="CAP40"; CAP50="CAP50"
        CAP60="CAP60"; CAP70="CAP70"; ET="ET"; HBR="HBR"; VIL="VIL"
    class MetNwp: 
        WS925="WS925"; WS700="WS700"; WS500="WS500"; Q1000="Q1000"; Q850="Q850"; Q700="Q700"
        PWAT="PWAT"; PE="PE"; TdSfc850="TdSfc850"; LCL="LCL"; KI="KI"; CAPE="CAPE"
    class MetGis: 
        LAT="LAT"; LON="LON"; DEM="DEM"; MONTH_SIN="MONTH_SIN"; MONTH_COS="MONTH_COS"; HOUR_SIN="HOUR_SIN"; HOUR_COS="HOUR_COS"

# ==========================================
# 1. 通道定义与属性
# ==========================================

# 基础30通道顺序 (必须与模型输入一致)
DEFAULT_BASE_CHANNELS = [
    MetLabel.RA, 
    MetRadar.CR, MetRadar.CAP20, MetRadar.CAP30, MetRadar.CAP40, MetRadar.CAP50, 
    MetRadar.CAP60, MetRadar.CAP70, MetRadar.ET, MetRadar.HBR, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS700, MetNwp.WS500, MetNwp.Q1000, MetNwp.Q850, 
    MetNwp.Q700, MetNwp.PWAT, MetNwp.PE, MetNwp.TdSfc850, MetNwp.LCL, MetNwp.KI, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH_SIN, MetGis.MONTH_COS, 
    MetGis.HOUR_SIN, MetGis.HOUR_COS
]

# NWP 变量列表 (用于 Folded 通道)
NWP_VAR_NAMES = ["WS925", "WS700", "WS500", "Q1000", "Q850", "Q700", "PWAT", "PE", "TdSfc850", "LCL", "KI", "CAPE"]

def get_channel_metadata(channel_name):
    """
    根据 met_var.py 的逻辑获取通道的 min, max, scale_factor, unit。
    """
    # 默认值
    c_min, c_max = 0.0, 1.0
    scale = 1.0
    unit = ""
    is_radar_ref = False # 是否为雷达反射率 (dBZ)
    is_precip = False    # 是否为降水

    # 1. 尝试从 MetVar 获取真实 min/max
    # 这里通过名字字符串反查，或者直接硬编码常见值以确保准确
    name = str(channel_name).split(':')[-1] # Remove prefix if any
    
    # === 特定逻辑 ===
    if name == "RA":
        c_min, c_max = 0, 300
        scale = 10.0 # 300 -> 30mm
        unit = "mm"
        is_precip = True
    elif name in ["CR", "HBR"] or name.startswith("CAP"):
        c_min, c_max = 0, 800
        scale = 10.0 # 800 -> 80dBZ
        unit = "dBZ"
        is_radar_ref = True
    elif name == "ET":
        c_min, c_max = 0, 150
        scale = 10.0 # 150 -> 15km
        unit = "km"
    elif name == "VIL":
        c_min, c_max = 0, 8000
        scale = 10.0 # 8000 -> 800? (按用户指示)
        unit = "kg/m²"
    elif name.startswith("WS"): # Wind Speed
        c_min, c_max = 0, 50 # 估算
        unit = "m/s"
    elif name.startswith("RH") or name == "PE":
        c_min, c_max = 0, 100
        unit = "%"
    elif name.startswith("Q"): # Specific Humidity
        c_min, c_max = 0, 30
        unit = "g/kg"
    elif name == "PWAT":
        c_min, c_max = 0, 80
        unit = "mm"
    elif name == "CAPE":
        c_min, c_max = 0, 3000
        unit = "J/kg"
    elif name == "DEM":
        c_min, c_max = 0, 3000
        unit = "m"
    elif "SIN" in name or "COS" in name:
        c_min, c_max = -1, 1
        unit = ""
    elif name in ["LAT", "LON"]:
        c_min, c_max = 0, 360 # 泛化
        unit = "°"
    
    return c_min, c_max, scale, unit, is_radar_ref, is_precip

# ==========================================
# 2. 色标定义
# ==========================================

def get_precip_cmap():
    """降水色标 (自定义)"""
    hex_colors = ['#3CB73A', '#63B7FF', '#0200F9', '#EE00F0', '#9F0000']
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_bad('white'); cmap.set_under('white')
    bounds = [0.1, 1, 2, 5, 8, 100]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(hex_colors))
    return cmap, norm

def get_radar_cmap():
    """雷达反射率色标 (dBZ Standard)"""
    colors = [
        '#E0F8F8', '#00ECEC', '#00A0F6', '#0000F6', '#00FF00', 
        '#00C800', '#009000', '#FFFF00', '#E7C000', '#FF9000', 
        '#FF0000', '#D60000', '#C00000', '#FF00FF'
    ]
    bounds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 85]
    cmap = mcolors.ListedColormap(colors)
    cmap.set_under('white')
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(colors))
    return cmap, norm

# ==========================================
# 3. 核心绘图函数
# ==========================================

def get_all_channel_names(in_seq=10, out_seq=20):
    """生成54个通道名称"""
    names = []
    # Base
    for c in DEFAULT_BASE_CHANNELS:
        val = c.value if hasattr(c, 'value') else str(c)
        names.append(val)
    # Folded NWP
    fold_factor = out_seq // in_seq 
    for f in range(fold_factor):
        for var_name in NWP_VAR_NAMES:
            names.append(f"Fold_T+{f+1}_{var_name}")
    return names

def visualize_sample_full(input_data, sample_idx, save_dir):
    """
    绘制 54 Channel x 10 Timestep 的大图 (修复顶部空白版)
    """
    # ... (数据转置逻辑不变) ...
    if input_data.shape[0] < input_data.shape[1]: 
        data_np = input_data.transpose(1, 0, 2, 3)
    else:
        data_np = input_data

    num_channels = data_np.shape[0]
    num_timesteps = data_np.shape[1]
    
    channel_names = get_all_channel_names()
    
    # 尺寸计算
    subplot_h, subplot_w = 1.0, 1.0
    fig_width = num_timesteps * subplot_w + 2.0 
    fig_height = num_channels * subplot_h
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # === 关键修复开始 ===
    # 动态计算 top 参数：保留固定的顶部空间（例如 0.5 英寸），而不是按比例
    # 0.5 英寸 / fig_height 即为需要的相对比例
    top_margin_inch = 0.5
    bottom_margin_inch = 0.2
    top_frac = 1.0 - (top_margin_inch / fig_height)
    bottom_frac = bottom_margin_inch / fig_height
    
    # 创建 GridSpec 时显式指定边界，消除默认的 10% 留白
    width_ratios = [1] * num_timesteps + [0.05] 
    gs = fig.add_gridspec(num_channels, num_timesteps + 1, 
                          width_ratios=width_ratios, 
                          wspace=0.05, hspace=0.1,
                          top=top_frac, bottom=bottom_frac, # <--- 强制贴顶
                          left=0.05, right=0.95)
    # === 关键修复结束 ===

    cmap_precip, norm_precip = get_precip_cmap()
    cmap_radar, norm_radar = get_radar_cmap()

    for c in range(num_channels):
        name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        c_min, c_max, scale, unit, is_radar, is_precip = get_channel_metadata(name)
        
        row_data_norm = data_np[c] 
        row_data_raw = row_data_norm * (c_max - c_min) + c_min
        row_data_phys = row_data_raw / scale

        if is_precip:
            cmap, norm = cmap_precip, norm_precip
            row_data_phys[row_data_phys < 0.1] = -1 
        elif is_radar:
            cmap, norm = cmap_radar, norm_radar
            row_data_phys[row_data_phys < 5] = -1 
        else:
            cmap = 'viridis'; norm = None

        for t in range(num_timesteps):
            ax = fig.add_subplot(gs[c, t])
            img = ax.imshow(row_data_phys[t], cmap=cmap, norm=norm, origin='lower')
            
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('black'); spine.set_linewidth(0.5)
            
            if t == 0:
                ax.set_ylabel(f"[{c}] {name}", rotation=0, ha='right', va='center', fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_title(f"T={t}", fontsize=10)

        # Colorbar 绘制逻辑不变...
        ax_cbar = fig.add_subplot(gs[c, -1])
        if norm:
            cbar = plt.colorbar(img, cax=ax_cbar, orientation='vertical', fraction=1.0)
            if is_precip: cbar.set_ticks([0.1, 1, 2, 5, 8])
            elif is_radar: cbar.set_ticks([10, 30, 50, 70])
        else:
            cbar = plt.colorbar(img, cax=ax_cbar, orientation='vertical')
            
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(unit, fontsize=8, rotation=270, labelpad=10)

    # 标题位置微调：放在计算出的 top 区域中间
    title_y = 1.0 - (top_margin_inch / 2 / fig_height)
    fig.suptitle(f'Sample {sample_idx} Input Visualization (Denormalized)', fontsize=16, y=title_y)
    
    save_path = os.path.join(save_dir, f"sample_{sample_idx}_input_real.png")
    print(f"[Info] Saving fixed visualization to: {save_path}")
    
    # 使用 bbox_inches='tight' 配合 pad_inches=0.1 裁剪多余白边
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    data_path = "data/samples.jsonl"
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)

    if not os.path.exists(data_path):
        print(f"[Error] 数据未找到: {data_path}")
        return
        
    print("[Info] Loading DataModule...")
    dm = ScwdsDataModule(
        data_path=data_path,
        batch_size=1,
        num_workers=0,
        train_split=0.99
    )
    dm.setup('fit')
    loader = dm.train_dataloader()

    samples_to_process = 10
    
    print(f"[Info] Processing first {samples_to_process} samples...")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= samples_to_process: break
        
        # input_batch shape: [B, T, C, H, W]
        _, input_batch, _, _, _ = batch
        
        # 取第一个样本
        input_data = input_batch[0].detach().cpu().numpy() # [T, C, H, W]
        
        visualize_sample_full(input_data, batch_idx, tmp_dir)
        
    print(f"[Success] Done. Check '{tmp_dir}'")

if __name__ == "__main__":
    main()