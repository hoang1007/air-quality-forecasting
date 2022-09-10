from typing import Dict, List, Optional
from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from .raw_data import private_train_data
from utils.functional import extract_wind


class PrivateDataModule(LightningDataModule):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        batch_size: int,
        train_ratio: float = 0.75
    ):
        super().__init__()
        self.rootdir = rootdir
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage: Optional[str] = None) -> None:
        datafull = PrivateDataset(self.rootdir, self.normalize_mean, self.normalize_std)

        train_size = int(len(datafull) * self.train_ratio)
        val_size = len(datafull) - train_size

        self.data_train, self.data_val = random_split(datafull, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=2)
    

class PrivateDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        data_set: str = "train"
    ):
        self.mean_ = normalize_mean
        self.std_ = normalize_std
        self.data_set = data_set
        self.inseq_len = 168
        self.outseq_len = 24

        if data_set == "train":
            self.data = private_train_data(path.join(rootdir, "data-train"))
        elif data_set == "test":
            raise NotImplementedError

    def __len__(self):
        if self.data_set == "train":
            return 243
        else:
            raise NotImplementedError
    
    def __getitem__(self, idx):
        if self.data_set == "train":
            return self._get_training_item(idx)
        else:
            raise NotImplementedError

    def _get_training_item(self, idx):
        in_air_ = []
        in_meteo_ = []
        air_locs_ = []
        meteo_locs_ = []
        targets_ = []

        in_s = idx * self.outseq_len
        in_e = in_s + self.inseq_len
        out_s = in_e
        out_e = out_s + self.outseq_len

        for station in self.data["air"].values():
            df = station["data"].iloc[in_s : in_e].copy()
            air = self._air_to_tensor(df, agg=False, norm=True)
            in_air_.append(air)

            target_df = station["data"].iloc[out_s : out_e].copy()
            target = self._air_to_tensor(target_df, usecols=["PM2.5"], norm=False).squeeze_(-1)
            targets_.append(target)

            air_locs_.append(station["loc"])

        meteo_s = idx * self.outseq_len // 3
        meteo_e = meteo_s + self.inseq_len // 3
        for station in self.data["meteo"].values():
            df = station["data"].iloc[meteo_s : meteo_e].copy()

            meteo = self._meteo_to_tensor(df, norm=True)
            in_meteo_.append(meteo)

            meteo_locs_.append(station["loc"])

        in_air_ = torch.stack(in_air_, dim=0)
        in_meteo_ = torch.stack(in_meteo_, dim=0)
        air_locs_ = torch.tensor(air_locs_)
        meteo_locs_ = torch.tensor(meteo_locs_)
        targets_ = torch.stack(targets_, dim=0)

        return {
            "air": in_air_,
            "meteo": in_meteo_,
            "air_locs": air_locs_,
            "meteo_locs": meteo_locs_,
            "targets": targets_,
        }
    
    def _air_to_tensor(
        self,
        df: pd.DataFrame,
        usecols: List[str] = ["humidity", "temperature", "PM2.5"],
        norm: bool = True,
        agg: bool = False
    ):
        out = []

        for col in usecols:
            assert df[col].isna().sum() == 0, "Data must not contains nan values"
            datacol = torch.tensor(df[col].values, dtype=torch.float32)

            if norm:
                datacol = (datacol - self.mean_[col]) / self.std_[col]

            out.append(datacol)

        # out.shape == (seq_len, n_cols)
        out = torch.stack(out, dim=-1)

        if agg:
            out = out.reshape(-1, 3, out.size(-1)).mean(1)

        return out

    def _meteo_to_tensor(
        self,
        df: pd.DataFrame,
        norm: bool = True
    ):
        # cols = ["Surface_pressure"]

        _, wind_spd = extract_wind(df["u10"].tolist(), df["v10"].tolist())
        wind_spd = torch.tensor(wind_spd, dtype=torch.float32)

        press = torch.tensor(df["surface_pressure"].values, dtype=torch.float32)

        if norm:
            wind_spd = (wind_spd - self.mean_["wind_speed"]) / self.std_["wind_speed"]
            press = (press - self.mean_["surface_pressure"]) / self.std_["surface_pressure"]

        return torch.stack((wind_spd, press), dim=-1)