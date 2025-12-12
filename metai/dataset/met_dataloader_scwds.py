import json
from typing import List, Dict, Any, Optional
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
from metai.dataset import MetSample
from metai.utils.met_config import get_config

class ScwdsDataset(Dataset):
    """
    SCWDS (Severe Convective Weather Dataset) 自定义数据集类。
    
    该类负责读取样本索引文件 (.jsonl)，并利用 MetSample 类加载具体的
    雷达、NWP 等多模态气象数据。
    """
    
    def __init__(self, data_path: str, is_train: bool = True, test_set: str = "TestSetB"):
        """
        初始化数据集。

        Args:
            data_path (str): 样本索引文件路径 (例如: 'data/samples.jsonl')。
            is_train (bool): 是否为训练/验证模式。True 会加载标签(Target)，False 仅加载输入。
            test_set (str): 测试集子目录名称 (例如: "TestSetB"), 用于构建文件路径。
        """
        self.data_path = data_path
        self.config = get_config()
        # 加载所有样本元数据
        self.samples = self._load_samples_from_jsonl(data_path)
        self.is_train = is_train
        self.test_set = test_set
        
    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本数据。
        
        Args:
            idx (int): 样本索引。
            
        Returns: 
            tuple: 包含以下元素的元组:
                - metadata (Dict): 样本元数据 (ID, 时间戳等)。
                - input_data (np.ndarray): 输入序列张量 (T_in, C, H, W)。
                - target_data (np.ndarray | None): 目标序列张量 (T_out, 1, H, W)。推理模式下为 None。
                - input_mask (np.ndarray): 输入掩码。
                - target_mask (np.ndarray | None): 目标掩码。推理模式下为 None。
        """
        record = self.samples[idx]
        
        # 创建 MetSample 实例来处理具体的文件读取和预处理
        sample = MetSample.create(
            record.get("sample_id"),
            record.get("timestamps"),
            config=self.config,
            is_train=self.is_train,
            test_set=self.test_set
        )
        
        # 调用 to_numpy 加载实际的数值数据
        return sample.to_numpy() 
                        
    def _load_samples_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 JSONL 文件中加载样本列表。

        Args:
            file_path (str): JSONL 文件路径。

        Returns:
            List[Dict[str, Any]]: 样本字典列表。
        """
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    samples.append(sample)
        return samples

class ScwdsDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule，用于管理 SCWDS 数据的加载、划分和批处理。
    
    功能包括：
    1. 自动划分 训练/验证/测试 集。
    2. 提供 train/val/test/infer 的 DataLoader。
    3. 自定义 collate_fn 以处理多模态数据的堆叠。
    """

    def __init__(
        self,
        data_path: str = "data/samples.jsonl",
        resize_shape: tuple[int, int] = (256, 256),
        aft_seq_length: int = 20,
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        初始化 DataModule。

        Args:
            data_path (str): 样本索引文件路径。
            resize_shape (tuple): 图片缩放的目标尺寸 (H, W)。
            aft_seq_length (int): 预测序列长度 (未使用，但在参数列表中保留)。
            batch_size (int): 批大小。
            num_workers (int): DataLoader 的工作线程数。
            pin_memory (bool): 是否将数据固定在 CUDA 内存中。
            train_split (float): 训练集比例。
            val_split (float): 验证集比例。
            test_split (float): 测试集比例。
            seed (int): 随机种子，保证数据集划分的可复现性。
        """
        super().__init__()
        self.data_path = data_path
        self.resize_shape = resize_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.original_shape = (301, 301)
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """
        根据当前阶段 (fit/test/infer) 准备数据集。
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', 或 'infer'。
        """
        # --- 推理模式 ---
        if stage == "infer":
            # 推理模式下，直接使用全量数据，无需分割，且 is_train=False
            self.infer_dataset = ScwdsDataset(
                self.data_path, 
                is_train=False
            )
            print(f"[INFO] Infer dataset size: {len(self.infer_dataset)}")
            return
        
        # --- 训练/验证/测试模式 ---
        if not hasattr(self, 'dataset'):
            self.dataset = ScwdsDataset(
                data_path=self.data_path,
                is_train=True
            )
            
            total_size = len(self.dataset)
            
            # 如果数据集为空或太小，跳过分割并发出警告
            if total_size == 0:
                print("[WARNING] Dataset is empty, skipping split")
                return
            
            # 计算各集合的具体数量
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            # 边界情况处理：确保训练集至少有一个样本
            if train_size == 0 and total_size > 0:
                train_size = 1
                test_size = total_size - train_size - val_size
            
            lengths = [train_size, val_size, test_size]

            # 创建确定性随机生成器，确保每次运行划分结果一致
            generator = torch.Generator().manual_seed(self.seed)

            # 随机划分数据集
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, lengths, generator=generator
            )
            
            print(f"[INFO] Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def _collate_fn(self, batch):
        """
        训练/验证/测试用的 Collate 函数。
        将 Dataset 返回的多个样本 (list of tuples) 堆叠成 Batch 张量。
        
        Returns:
            metadata_batch (List): 元数据列表。
            input_batch (Tensor): [B, T_in, C, H, W]
            target_batch (Tensor): [B, T_out, 1, H, W]
            target_mask_batch (Tensor): [B, T_out, 1, H, W]
            input_mask_batch (Tensor): [B, T_in, C, H, W]
        """
        metadata_batch = []
        input_tensors = []
        target_tensors = []
        target_mask_tensors = []
        input_mask_tensors = []

        for metadata, input_np, target_np, input_mask_np, target_mask_np in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.from_numpy(input_np).float())
            target_tensors.append(torch.from_numpy(target_np).float())
            target_mask_tensors.append(torch.from_numpy(target_mask_np).bool())
            input_mask_tensors.append(torch.from_numpy(input_mask_np).bool())

        # 使用 torch.stack 将列表转换为张量，并在维度 0 (Batch) 上堆叠
        input_batch = torch.stack(input_tensors, dim=0).contiguous()
        target_batch = torch.stack(target_tensors, dim=0).contiguous()
        target_mask_batch = torch.stack(target_mask_tensors, dim=0).contiguous()
        input_mask_batch = torch.stack(input_mask_tensors, dim=0).contiguous()
        
        return metadata_batch, input_batch, target_batch, input_mask_batch, target_mask_batch

    def _collate_fn_infer(self, batch):
        """
        推理用的 Collate 函数。
        与 _collate_fn 的区别在于不处理 target (标签) 数据。
        
        Returns:
            metadata_batch (List): 元数据列表。
            input_batch (Tensor): [B, T_in, C, H, W]
            input_mask_batch (Tensor): [B, T_in, C, H, W]
        """
        metadata_batch = []
        input_tensors = []
        input_mask_tensors = []

        for metadata, input_np, _, input_mask_np, _ in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.from_numpy(input_np).float())
            input_mask_tensors.append(torch.from_numpy(input_mask_np).bool())
        
        input_batch = torch.stack(input_tensors, dim=0).contiguous()
        input_mask_batch = torch.stack(input_mask_tensors, dim=0).contiguous()
        
        return metadata_batch, input_batch, input_mask_batch


    def train_dataloader(self):
        """返回训练数据加载器"""
        if not hasattr(self, 'train_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # 训练集需要打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """返回验证数据加载器"""
        if not hasattr(self, 'val_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # 验证集不需要打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """返回测试数据加载器"""
        if not hasattr(self, 'test_dataset'):
            self.setup('test')
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def infer_dataloader(self) -> Optional[DataLoader]:
        """返回推理数据加载器"""
        if not hasattr(self, 'infer_dataset'):
            self.setup('infer')
            
        return DataLoader(
            self.infer_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn_infer # 推理模式使用专门的 collate_fn (无标签)
        )