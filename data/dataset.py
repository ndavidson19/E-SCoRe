import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from .dataset_generator import SATQuestion

class StructuredSATDataset(Dataset):
    def __init__(self, data: List[SATQuestion]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(dataset: StructuredSATDataset, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)