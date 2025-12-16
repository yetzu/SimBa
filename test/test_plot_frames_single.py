import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    # 尝试导入真实常量 (MetLabel 等)
    from metai.utils.met_var import MetLabel, MetRadar, MetNwp, MetGis
except ImportError:
    print("[Warning] 无法导入 metai.utils.met_var，使用内置定义。")
    # Fallback Definitions
    class MetLabel: RA = "RA"
    class MetRadar: 
        CR="CR"; CAP20="CAP20"; CAP30="CAP30"; CAP40="CAP40"; CAP50="CAP50"
        CAP60="CAP60"; CAP70="CAP70"; ET="ET"; HBR="HBR"; VIL="VIL"
    class MetNwp: 
        WS925="WS925"; WS700="WS700"; WS500="WS500"; Q1000="Q1000"; Q850="Q850"; Q700="Q700"
        PWAT="PWAT"; PE="PE"; TdSfc850="TdSfc850"; LCL="LCL"; KI="KI"; CAPE="CAPE"
    class MetGis: 
        LAT="LAT"; LON="LON"; DEM="DEM"; MONTH_SIN="MONTH_SIN"; MONTH_COS="MONTH_COS"; HOUR_SIN="HOUR_SIN"; HOUR_COS="HOUR_COS"

# ==========================================
# 1. 通道与属性定义
# ==========================================
DEFAULT_BASE_CHANNELS = [
    MetLabel.RA, 
    MetRadar.CR, MetRadar.CAP20, MetRadar.CAP30, MetRadar.CAP40, MetRadar.CAP50, 
    MetRadar.CAP60, MetRadar.CAP70, MetRadar.ET, MetRadar.HBR, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS700, MetNwp.WS500, MetNwp.Q1000, MetNwp.Q850, 
    MetNwp.Q700, MetNwp.PWAT, MetNwp.PE, MetNwp.TdSfc850, MetNwp.LCL, MetNwp.KI, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH_SIN, MetGis.MONTH_COS, 
    MetGis.HOUR_SIN, MetGis.HOUR_COS
]
NWP_VAR_NAMES = ["WS925", "WS700", "WS500", "Q1000", "Q850", "Q700", "PWAT", "PE", "TdSfc850", "LCL", "KI", "CAPE"]

def get_all_channel_names(in_seq=10, out_seq=20):
    """生成所有通道名称"""
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

def get_channel_metadata(channel_name):
    """反归一化配置 (min, max, scale)"""
    c_min, c_max = 0.0, 1.0
    scale = 1.0
    is_radar_ref = False
    is_precip = False

    name = str(channel_name).split(':')[-1] # Remove prefix
    
    if name == "RA":
        c_min, c_max = 0, 300
        scale = 10.0
        is_precip = True
    elif name in ["CR", "HBR"] or name.startswith("CAP"):
        c_min, c_max = 0, 800
        scale = 10.0
        is_radar_ref = True
    elif name == "ET":
        c_min, c_max = 0, 150
        scale = 10.0
    elif name == "VIL":
        c_min, c_max = 0, 8000
        scale = 10.0
    elif name.startswith("WS"): c_min, c_max = 0, 50
    elif name.startswith("RH") or name == "PE": c_min, c_max = 0, 100
    elif name.startswith("Q"): c_min, c_max = 0, 30
    elif name == "PWAT": c_min, c_max = 0, 80
    elif name == "CAPE": c_min, c_max = 0, 3000
    elif name == "DEM": c_min, c_max = 0, 3000
    elif "SIN" in name or "COS" in name: c_min, c_max = -1, 1
    elif name in ["LAT", "LON"]: c_min, c_max = 0, 360
    
    return c_min, c_max, scale, is_radar_ref, is_precip

# ==========================================
# 2. 色标定义
# ==========================================
def get_precip_cmap():
    hex_colors = ['#3CB73A', '#63B7FF', '#0200F9', '#EE00F0', '#9F0000']
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_bad('white'); cmap.set_under('white')
    bounds = [0.1, 1, 2, 5, 8, 100]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(hex_colors))
    return cmap, norm

def get_radar_cmap():
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
# 3. 核心绘图逻辑
# ==========================================
def save_single_frame(data_frame, save_path, cmap, norm=None, vmin=None, vmax=None):
    """保存单张图片：无标题、无图例、有边框"""
    # 设定图像大小 (例如 3x3 inches)，DPI=100 -> 300x300 pixels
    # 这里的 figsize 可以根据需要调整，bbox_inches='tight' 会自动裁剪
    fig = plt.figure(figsize=(3, 3))
    
    # 创建填充整个画布的 axes
    # [left, bottom, width, height] = [0, 0, 1, 1] 确保图像撑满，但我们需要边框
    # 所以使用 add_subplot 即可，后续去掉 axis
    ax = fig.add_subplot(111)
    
    # 绘图
    ax.imshow(data_frame, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, origin='lower')
    
    # 设置边框
    # 隐藏刻度
    ax.set_xticks([])
    ax.set_yticks([])
    # 显式开启边框线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0) # 边框稍微加粗以便观察

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # bbox_inches='tight', pad_inches=0 -> 移除所有外部空白
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_sample(input_data, sample_idx, output_root):
    """处理单个样本的所有通道和时间步"""
    
    # 确保数据维度是 [C, T, H, W]
    if input_data.shape[0] < input_data.shape[1]: 
        # 假设输入是 [T, C, H, W] (10, 54, ...)，转置为 [C, T, ...]
        data_np = input_data.transpose(1, 0, 2, 3)
    else:
        data_np = input_data

    num_channels = data_np.shape[0]
    num_timesteps = data_np.shape[1]
    channel_names = get_all_channel_names()
    
    cmap_precip, norm_precip = get_precip_cmap()
    cmap_radar, norm_radar = get_radar_cmap()

    print(f"[Info] Generating frames for Sample {sample_idx}...")

    for c in range(num_channels):
        ch_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        
        # 1. 获取配置
        c_min, c_max, scale, is_radar, is_precip = get_channel_metadata(ch_name)
        
        # 2. 反归一化
        row_data_norm = data_np[c] 
        row_data_raw = row_data_norm * (c_max - c_min) + c_min
        row_data_phys = row_data_raw / scale
        
        # 3. 确定绘图参数
        cmap = 'viridis' # 默认
        norm = None
        vmin, vmax = None, None

        if is_precip:
            cmap, norm = cmap_precip, norm_precip
            row_data_phys[row_data_phys < 0.1] = -1 # 设置背景
        elif is_radar:
            cmap, norm = cmap_radar, norm_radar
            row_data_phys[row_data_phys < 5] = -1 # 设置背景
        else:
            cmap = 'jet'
            # 对于连续变量，直接使用物理范围作为 clim，或者让 imshow 自动处理
            # 这里如果不设 vmin/vmax，Matplotlib 会自动 scale 到每帧的 min-max，这可能导致时间序列闪烁
            # 建议不设，或者设为物理极值
            # vmin = c_min / scale
            # vmax = c_max / scale
            pass

        # 4. 循环时间步保存
        for t in range(num_timesteps):
            # 路径: tmp/single/sample_{id}/{CHANNEL_NAME}/{t}.png
            # t:02d -> 00, 01, 02...
            save_path = os.path.join(output_root, f"sample_{sample_idx}", ch_name, f"{t:02d}.png")
            
            save_single_frame(row_data_phys[t], save_path, cmap, norm, vmin, vmax)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    data_path = "data/samples.jsonl"
    tmp_dir = "tmp/single" # 输出根目录
    
    if not os.path.exists(data_path):
        print(f"[Error] 数据未找到: {data_path}")
        return
        
    print("[Info] Loading DataModule...")
    # num_workers=0 避免多进程带来的调试困难
    dm = ScwdsDataModule(
        data_path=data_path,
        batch_size=1, 
        num_workers=0,
        train_split=0.99
    )
    dm.setup('fit')
    loader = dm.train_dataloader()

    # 处理样本数量
    samples_to_process = 10
    
    data_iter = iter(loader)
    for i in range(samples_to_process):
        try:
            batch = next(data_iter)
            # input_batch shape: [B, T, C, H, W]
            _, input_batch, _, _, _ = batch
            
            # 取 batch 中第一个样本
            input_data = input_batch[0].detach().cpu().numpy()
            
            process_sample(input_data, i, tmp_dir)
            
        except StopIteration:
            break
            
    print(f"[Success] All frames saved to '{os.path.abspath(tmp_dir)}'")

if __name__ == "__main__":
    main()