import os

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

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


def get_dataloader(args, batch_size, seed):

    train_path = os.path.join(args.data_dir, 'train.h5')
    val_path = os.path.join(args.data_dir, 'val.h5')
    test_path = os.path.join(args.data_dir, 'test_x_only.h5')

    #获取数据
    train_data = TrainDataset(train_path)
    val_data = TrainDataset(val_path)   
    test_data = TestDataset(test_path)

    print("data shape:")
    print(f"  train X: {tuple(train_data.x.shape)}, y: {tuple(train_data.y.shape)}")
    print(f"  val   X: {tuple(val_data.x.shape)}, y: {tuple(val_data.y.shape)}")
    print(f"  test  X: {tuple(test_data.x.shape)}")

    train_loader = DataLoader(train_data,   batch_size = batch_size,    shuffle=True)
    val_loader = DataLoader(val_data,       batch_size = batch_size,    shuffle=False)
    test_loader = DataLoader(test_data,     batch_size = 4,             shuffle=False)

    return train_loader, val_loader, test_loader

