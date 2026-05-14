import os

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from util import normalize_data

# --- Train: 读 x 和 y ---
class TrainDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)
        print(self.x.shape)
        assert len(self.x) == len(self.y), "X and y length mismatch"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- Test: 只读 x ---
class TestDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def print_data_stats(split_name, tensor_x):
    """
    计算并打印张量的最大值、最小值、均值和标准差
    """
    max_val = tensor_x.max().item()
    min_val = tensor_x.min().item()
    mean_val = tensor_x.mean().item()
    std_val = tensor_x.std().item()
    
    print(f"  {split_name:<7} | Max: {max_val:>8.4f} | Min: {min_val:>8.4f} | Mean: {mean_val:>8.4f} | Std: {std_val:>8.4f}")

def get_dataloader(args, batch_size, seed, dataset_name):

    train_path = os.path.join(args.data_dir, 'train.h5')
    val_path = os.path.join(args.data_dir, 'val.h5')
    test_path = os.path.join(args.data_dir, 'test_x_only.h5')

    #获取数据
    train_data = TrainDataset(train_path)
    val_data = TrainDataset(val_path)   
    test_data = TestDataset(test_path)

    # 数据缩放
    use_z = False                   # 是否直接使用Z-score标准化（如果False，则使用缩放）
    train_data.x = normalize_data(train_data.x, dataset_name, use_zscore=use_z)
    val_data.x = normalize_data(val_data.x, dataset_name, use_zscore=use_z)
    test_data.x = normalize_data(test_data.x, dataset_name, use_zscore=use_z)


    print("\n" + "="*60)
    print("📊 数据统计信息 (Raw Data Statistics)")
    print("-" * 60)
    print(f"  train X: {tuple(train_data.x.shape)}, y: {tuple(train_data.y.shape)}")
    print(f"  val   X: {tuple(val_data.x.shape)}, y: {tuple(val_data.y.shape)}")
    print(f"  test  X: {tuple(test_data.x.shape)}")
    # print(f"y label :  {tuple(val_data.y)}")
    print("-" * 60)     
    # 调用统计函数
    print_data_stats("Train", train_data.x)
    print_data_stats("Val",   val_data.x)
    print_data_stats("Test",  test_data.x)
    print("="*60 + "\n")


    train_loader = DataLoader(train_data,   batch_size = batch_size,    shuffle=True)
    val_loader = DataLoader(val_data,       batch_size = batch_size,    shuffle=False)
    test_loader = DataLoader(test_data,     batch_size = 4,             shuffle=False)

    return train_loader, val_loader, test_loader

