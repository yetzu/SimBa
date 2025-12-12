import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import ScwdsDataset

def resize_tensor(data_np: np.ndarray, target_shape: tuple, mode: str = 'max_pool') -> np.ndarray:
    """
    参考 _interpolate_batch_gpu 实现的独立插值函数。
    将 Numpy (T, C, H, W) -> Torch (1, T, C, H, W) -> Resize -> Numpy (T, C, H', W')
    """
    if data_np is None:
        return None
        
    # 1. 转换为 Tensor 并增加 Batch 维度 (T, C, H, W) -> (1, T, C, H, W)
    tensor = torch.from_numpy(data_np).unsqueeze(0)
    
    # 2. 获取维度信息
    B, T, C, H, W = tensor.shape
    target_H, target_W = target_shape
    
    # 如果尺寸一致，直接返回原数据
    if H == target_H and W == target_W:
        return data_np

    # 3. 处理布尔类型 (Mask)
    is_bool = tensor.dtype == torch.bool
    if is_bool:
        tensor = tensor.float()
    
    # 4. 合并 B 和 T 维度 (B*T, C, H, W)
    tensor = tensor.view(B * T, C, H, W)
    
    # 5. 执行插值
    if mode == 'max_pool':
        # 下采样使用 adaptive_max_pool2d 保留极值
        if target_H < H or target_W < W:
            processed = F.adaptive_max_pool2d(tensor, output_size=target_shape)
        else:
            processed = F.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    elif mode in ['nearest', 'bilinear']:
        align = False if mode == 'bilinear' else None
        processed = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=align)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    
    # 6. 还原维度 (1, T, C, H', W') -> (T, C, H', W')
    processed = processed.view(B, T, C, target_H, target_W)
    if is_bool:
        processed = processed.bool()
        
    return processed.squeeze(0).numpy()

def export_train_data_to_npz(data_path, output_dir, target_size=(256, 256)):
    """
    遍历 ScwdsDataset (训练模式)，下采样并保存为 .npz 文件。
    """
    
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] 创建输出目录: {output_dir}")

    # 2. 初始化数据集 (强制为训练模式)
    print(f"[INFO] 正在初始化数据集 (Mode: Train)...")
    try:
        ds = ScwdsDataset(
            data_path=data_path, 
            is_train=True 
        )
    except Exception as e:
        print(f"[ERROR] 数据集初始化失败: {e}")
        return

    print(f"[INFO] 共发现 {len(ds)} 个样本，准备导出并Resize到 {target_size}...")

    # 3. 遍历并保存
    success_count = 0
    fail_count = 0

    # NOTE: torch Dataset 不保证实现 Iterable；为兼容类型检查与所有 Dataset 实现，这里用索引遍历。
    for i in tqdm(range(len(ds)), desc="Exporting & Resizing"):
        batch = ds[i]
        metadata = {}
        try:
            # 解包 (匹配 to_numpy 修改后的顺序)
            metadata, input_data, target_data, input_mask, target_mask = batch
            
            # input_data: (T, C, H, W) -> max_pool -> (T, C, 256, 256)
            input_data_resized = resize_tensor(input_data, target_size, mode='max_pool')
            
            # target_data: (T, 1, H, W) -> max_pool -> (T, 1, 256, 256)
            target_data_resized = resize_tensor(target_data, target_size, mode='max_pool')
            
            # target_mask: (T, 1, H, W) -> nearest -> (T, 1, 256, 256)
            target_mask_resized = resize_tensor(target_mask, target_size, mode='nearest')
            
            # -----------------------------

            # 获取样本ID作为文件名
            sample_id = metadata.get('sample_id', f'sample_{i}')
            save_path = os.path.join(output_dir, f"{sample_id}.npz")
            
            # 仅保存指定的三个字段
            save_dict = {
                "input_data": input_data_resized,
                "target_data": target_data_resized,
                "target_mask": target_mask_resized
            }
            
            # 保存为压缩文件
            np.savez_compressed(save_path, **save_dict)
            success_count += 1
            
        except Exception as e:
            sid = metadata.get('sample_id', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            print(f"\n[WARNING] 样本 {i} (ID: {sid}) 处理失败: {e}")
            fail_count += 1

    print(f"\n[DONE] 处理结束。")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"文件保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 ScwdsDataset 导出为 .npz (Resize to 256x256)")
    
    parser.add_argument("--data_path", type=str, default="data/samples.jsonl", 
                        help="样本索引文件路径")
    parser.add_argument("--output_dir", type=str, default="/data/zjobs/SevereWeather_AI_2025/CP/Train", 
                        help="结果保存目录")

    args = parser.parse_args()
    
    export_train_data_to_npz(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_size=(256, 256)
    )