# prework/step6_export_dataset_to_npz.py
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing

# 确保能导入项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import ScwdsDataset

# --- 全局变量，用于子进程 ---
worker_ds = None 
global_output_dir = None
global_target_size = None

def resize_tensor(data_np: np.ndarray, target_shape: tuple, mode: str = 'max_pool') -> np.ndarray:
    """独立的插值函数 (逻辑保持不变)"""
    if data_np is None:
        return None
    tensor = torch.from_numpy(data_np).unsqueeze(0)
    B, T, C, H, W = tensor.shape
    target_H, target_W = target_shape
    if H == target_H and W == target_W:
        return data_np
    is_bool = tensor.dtype == torch.bool
    if is_bool:
        tensor = tensor.float()
    tensor = tensor.view(B * T, C, H, W)
    
    if mode == 'max_pool':
        if target_H < H or target_W < W:
            processed = F.adaptive_max_pool2d(tensor, output_size=target_shape)
        else:
            processed = F.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    elif mode in ['nearest', 'bilinear']:
        align = False if mode == 'bilinear' else None
        processed = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=align)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    
    processed = processed.view(B, T, C, target_H, target_W)
    if is_bool:
        processed = processed.bool()
    return processed.squeeze(0).numpy()

def worker_init(data_path, output_dir, target_size):
    """
    子进程初始化函数。
    每个子进程只执行一次：初始化 Dataset，设置全局变量。
    """
    global worker_ds, global_output_dir, global_target_size
    
    # 限制 Torch 在子进程中只用单核，避免与多进程冲突导致效率下降
    torch.set_num_threads(1)
    
    # 初始化 Dataset
    # 注意：这里假设 ScwdsDataset 读取 JSONL 很快。
    # 如果 JSONL 巨大，可以考虑只传索引，主进程读 JSONL (稍微复杂点)，
    # 但通常这里直接 init 是最简单的方案。
    try:
        worker_ds = ScwdsDataset(data_path=data_path, is_train=True)
    except Exception as e:
        print(f"[Worker Error] Dataset init failed: {e}")
        worker_ds = None

    global_output_dir = output_dir
    global_target_size = target_size

def process_sample(idx):
    """
    单个样本的处理逻辑。
    由子进程调用。
    """
    global worker_ds, global_output_dir, global_target_size
    
    if worker_ds is None:
        return False, f"Dataset not initialized in worker"

    try:
        batch = worker_ds[idx]
        metadata, input_data, target_data, input_mask, target_mask = batch
        
        # --- Resize ---
        input_data_resized = resize_tensor(input_data, global_target_size, mode='max_pool')
        target_data_resized = resize_tensor(target_data, global_target_size, mode='max_pool')
        target_mask_resized = resize_tensor(target_mask, global_target_size, mode='nearest')
        
        # --- Save ---
        sample_id = metadata.get('sample_id', f'sample_{idx}')
        save_path = os.path.join(global_output_dir, f"{sample_id}.npz")
        
        save_dict = {
            "input_data": input_data_resized,
            "target_data": target_data_resized,
            "target_mask": target_mask_resized
        }
        
        # np.savez_compressed(save_path, **save_dict)
        np.savez(save_path, **save_dict)
        return True, None
        
    except Exception as e:
        sid = 'Unknown'
        if 'metadata' in locals():
            sid = metadata.get('sample_id', 'Unknown')
        return False, f"Sample {idx} (ID: {sid}) failed: {e}"

def export_train_data_parallel(data_path, output_dir, target_size=(256, 256), num_workers=8):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] 创建输出目录: {output_dir}")

    # 2. 预先读取一次 Dataset 长度 (为了 tqdm 和 range)
    # 只需要轻量级初始化一次，或者手动解析 jsonl 行数
    print("[INFO] 正在计算样本总数...")
    temp_ds = ScwdsDataset(data_path=data_path, is_train=True)
    total_samples = len(temp_ds)
    del temp_ds # 释放
    print(f"[INFO] 样本总数: {total_samples}")
    print(f"[INFO] 启动并行处理 (Workers: {num_workers})...")

    # 3. 配置多进程 Pool
    # 这里的 initializer 会确保每个 worker 都有自己的 dataset 实例
    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(data_path, output_dir, target_size)
    )

    # 4. 并行执行
    success_count = 0
    fail_count = 0
    
    # 使用 imap_unordered 可以让结果无序返回（处理完一个返回一个），让进度条更丝滑
    # chunksize 可以适当调大，减少进程间通信频率
    results = list(tqdm(
        pool.imap_unordered(process_sample, range(total_samples), chunksize=10),
        total=total_samples,
        desc="Exporting"
    ))

    # 5. 关闭 Pool
    pool.close()
    pool.join()

    # 6. 统计结果
    for success, msg in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            if msg: print(f"\n[WARNING] {msg}")

    print(f"\n[DONE] 处理结束。")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"文件保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行导出 ScwdsDataset 为 .npz")
    
    parser.add_argument("--data_path", type=str, default="data/samples.jsonl", 
                        help="样本索引文件路径")
    parser.add_argument("--output_dir", type=str, default="/data/zjobs/SevereWeather_AI_2025/CP/Train", 
                        help="结果保存目录")
    # 默认 worker 数量设为 CPU 核心数，可根据实际情况调整
    parser.add_argument("--num_workers", type=int, default=50, 
                        help="并行进程数")

    args = parser.parse_args()
    
    # Windows 下 multiprocessing 必须在 if __name__ == '__main__': 保护下运行
    export_train_data_parallel(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_size=(256, 256),
        num_workers=args.num_workers
    )