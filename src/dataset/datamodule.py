from os import path
from typing import Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from datetime import datetime
from .raw_data import air_quality_train_data, air_quality_test_data
from utils.functional import get_solar_term, get_next_period

def default_fillnan_fn(x: pd.Series):
    return x.interpolate(option="linear")

class AirQualityDataModule(LightningDataModule):
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
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        datafull = AirQualityDataset(
            self.rootdir,
            self.normalize_mean,
            self.normalize_std,
            data_set="train"
        )

        train_size = int(len(datafull) * self.train_ratio)
        val_size = len(datafull) - train_size

        self.train_data, self.val_data = random_split(datafull, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=1)


class AirQualityDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        data_set: str = "train",
    ):
        super().__init__()

        self.data_set = data_set

        self.inseq_len = 24 * 7
        self.outseq_len = 24
        self.mean_ = normalize_mean
        self.std_ = normalize_std

        if data_set == "train":
            self.data = air_quality_train_data(
                path.join(rootdir, "data-train")
            )

            self._prev_fill_nan()
        elif data_set == "test":
            self.data = air_quality_test_data(
                path.join(rootdir, "public-test"),
                path.join(rootdir, "data-train")
            )

    def __len__(self):
        if self.data_set == "train":
            return 368
        elif self.data_set == "test":
            return 100
        else:
            raise ValueError

    def __getitem__(self, idx):
        if self.data_set == "train":
            return self._get_training_item(idx)
        elif self.data_set == "test":
            return self._get_testing_item(idx)
        else:
            raise ValueError

    def _get_training_item(self, idx):
        inputs = []
        in_locs = []
        src_nexts = []
        time = None

        in_start_idx = idx * self.outseq_len
        in_end_idx = in_start_idx + self.inseq_len
        out_start_idx = in_end_idx
        out_end_idx = out_start_idx + self.outseq_len

        for station in self.data["input"].values():
            df = station["data"].iloc[in_start_idx : in_end_idx].copy()
            next_df = station["data"].iloc[out_start_idx : out_end_idx]

            metero = self._metero_to_tensor(df, norm=True)
            pm25_next = self._metero_to_tensor(next_df, usecols=["PM2.5"], norm=False).squeeze_(-1)

            if time is None:
                time = self._time_to_tensor(df)

            inputs.append(metero)
            in_locs.append(torch.tensor(station["loc"], dtype=torch.float))
            src_nexts.append(pm25_next)

        targets = []
        tar_locs = []
        for station in self.data["output"].values():
            df = station["data"].iloc[out_start_idx : out_end_idx]

            pm25 = self._metero_to_tensor(df, usecols=["PM2.5"], norm=False).squeeze_(-1)
            
            targets.append(pm25)
            tar_locs.append(torch.tensor(station["loc"], dtype=torch.float))

        return {
            "metero": torch.stack(inputs, dim=0),
            "time": time,
            "src_nexts": torch.stack(src_nexts, dim=0),
            "src_locs": torch.stack(in_locs, dim=0),
            "targets": torch.stack(targets, dim=0),
            "tar_locs": torch.stack(tar_locs, dim=0),
        }

    def _get_testing_item(self, idx):
        dt = self.data["input"][idx]
        
        inputs = []
        in_locs = []
        time = None

        for station in dt.values():
            df = station["data"]

            metero = self._metero_to_tensor(df)
            inputs.append(metero)

            if time is None:
                time = self._time_to_tensor(df)

            in_locs.append(torch.tensor(station["loc"], dtype=torch.float))

        tar_locs = list(self.data["loc_output"].values())

        return {
            "metero": torch.stack(inputs, dim=0),
            "time": time,
            "src_locs": torch.stack(in_locs, dim=0),
            "tar_locs": torch.tensor(tar_locs, dtype=torch.float),
            "folder_name": self.data["folder_name"][idx]
        }

    def _prev_fill_nan(self):
        for k in ("input", "output"):
            for station in self.data[k].values():
                station["data"] = default_fillnan_fn(station["data"])

    def _metero_to_tensor(
        self,
        df: pd.DataFrame,
        usecols: List[str] = ["humidity", "temperature", "PM2.5"],
        norm: bool = True
    ):
        out = []

        for col in usecols:
            assert df[col].isna().sum() == 0, "Data must not contains nan values"
            datacol = torch.tensor(df[col].values, dtype=torch.float32)

            if norm:
                datacol = (datacol - self.mean_[col]) / self.std_[col]

            out.append(datacol)

        return torch.stack(out, dim=-1)

    def _time_to_tensor(
        self,
        df: pd.DataFrame
    ):
        time = {
            "hour": [],
            "weekday": [],
            "solar_term": [],
        }

        for date in df["timestamp"]:
            date = datetime.strptime(date, "%d/%m/%Y %H:%M")

            time["hour"].append(date.hour)
            time["weekday"].append(date.isoweekday())
            time["solar_term"].append(get_solar_term(date))

        for k in time:
            time[k] = torch.tensor(time[k], dtype=torch.long)

        return time