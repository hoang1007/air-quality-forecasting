from typing import Dict
from os import path
import torch
from torch.utils.data import Dataset, DataLoader
from .raw_data import private_train_data


class PrivateDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        data_set: str = "train"
    ):
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.data_set = data_set

        if data_set == "train":
            self.data = private_train_data(path.join())
