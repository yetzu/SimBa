import unittest
import os
import sys
import json
import shutil
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils.met_config import get_config
from metai.dataset.met_dataloader_scwds import ScwdsDataset, ScwdsDataModule


class TestScwdsRealData(unittest.TestCase):
    def setUp(self):
        """测试环境搭建"""
        self.test_dir = tempfile.mkdtemp()
        self.temp_jsonl_path = os.path.join(self.test_dir, 'samples.jsonl')
        
        # 1. 定位真实的 data/samples.jsonl 文件
        # 尝试常见路径：当前目录 或 父级目录
        possible_paths = [
            'data/samples.jsonl',           # Run from root
            '../data/samples.jsonl',        # Run from prework/ or tests/
            os.path.join(os.path.dirname(__file__), '../data/samples.jsonl') # Run relative to script
        ]
        
        real_jsonl_path = None
        for p in possible_paths:
            if os.path.exists(p):
                real_jsonl_path = p
                break
        
        if real_jsonl_path is None:
            raise FileNotFoundError("无法找到 data/samples.jsonl，请确保你在项目根目录或子目录下运行测试。")

        print(f"\n[Setup] Reading real data from: {os.path.abspath(real_jsonl_path)}")

        # 2. 读取前2行并写入临时文件
        count = 0
        with open(real_jsonl_path, 'r', encoding='utf-8') as f_in, \
             open(self.temp_jsonl_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if not line.strip(): continue
                f_out.write(line)
                count += 1
                if count >= 5:
                    break
        
        print(f"[Setup] Created temporary test file with {count} samples: {self.temp_jsonl_path}")

        # 打印配置信息
        config = get_config()
        print(f"[Setup] Config Root Path: {config.root_path}")

    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_dataset_loading(self):
        """测试 Dataset 从原始文件加载 (npz_dir=None)"""
        print(f"\n[Test 1] Testing Dataset Loading...")
        
        # 强制 npz_dir=None 以测试原始 MetSample 读取逻辑
        ds = ScwdsDataset(data_path=self.temp_jsonl_path, is_train=True)
        
        self.assertGreater(len(ds), 0, "数据集应该是非空的")
        
        # 读取第一个样本
        try:
            item = ds[0]
        except Exception as e:
            self.fail(f"Dataset.__getitem__ failed: {e}")

        # 解包: record, input, target, input_mask, target_mask
        _, input_data, target_data, input_mask, target_mask = item
        
        print(f"  -> Input Shape: {input_data.shape}")
        
        # 验证: Input 应该是 (T, C, 256, 256)
        # 注意：因为之前的 DataModule 移除了 Resize，且这里是从 MetSample 直接读取，所以是 256
        self.assertEqual(input_data.ndim, 4)
        self.assertEqual(input_data.shape[-2:], (256, 256), "原始数据尺寸应为 256x256")
        
        # 简单的数据有效性检查 (是否全0)
        if input_data.max() == 0 and input_data.min() == 0:
            print("  -> [WARNING] Input data is all Zeros! (Are the radar files actually in config.root_path?)")
        else:
            print(f"  -> [SUCCESS] Data loaded. Mean value: {input_data.mean():.4f}")

    def test_02_datamodule_batching(self):
        """测试 DataModule 批量处理"""
        print("\n[Test 2] Testing DataModule Batching...")
        
        dm = ScwdsDataModule(
            data_path=self.temp_jsonl_path,
            batch_size=2,
            num_workers=0
        )
        dm.setup('fit')
        
        # 检查训练集大小
        train_size = len(dm.train_dataset)
        print(f"  -> Train Dataset Size: {train_size}")
        if train_size < 2:
            self.skipTest(f"样本数量不足 ({train_size})，跳过 BatchSize=2 的测试")
            
        loader = dm.train_dataloader()
        
        try:
            batch = next(iter(loader))
        except StopIteration:
            self.fail("DataLoader is empty")
        except Exception as e:
            self.fail(f"DataLoader iteration failed: {e}")
            
        # 解包 Batch
        metadata_batch, input_batch, target_batch, input_mask_batch, target_mask_batch = batch
        
        # 验证 Batch 维度
        self.assertEqual(input_batch.shape[0], 2, "Batch Size 应为 2")
        self.assertEqual(input_batch.shape[-2:], (256, 256), "Batch 尺寸应保持 256x256")
        
        print(f"  -> Batch Input Shape: {input_batch.shape}")
        print("  -> Batching OK.")

if __name__ == '__main__':
    unittest.main()