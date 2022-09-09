from typing import List, Optional
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from .raw_data import public_train_data
from utils.functional import get_solar_term


def default_fillnan_fn(x: pd.Series):
    return x.interpolate(option="spline").bfill().ffill()


class BejingDataModule(LightningDataModule):
    def __init__(
        self,
        rootdir: str,
        batch_size: int,
        train_ratio: float = 0.75
    ):
        super().__init__()
        
        self.rootdir = rootdir
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage: Optional[str] = None) -> None:
        datafull = BejingDataset(self.rootdir)

        train_size = int(len(datafull) * self.train_ratio)
        val_size = len(datafull) - train_size

        self.train_data, self.val_data = random_split(datafull, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=1)


class BejingDataset(Dataset):
    def __init__(self, rootdir: str):
        self.inseq_len = 24 * 7
        self.outseq_len = 24
        self.metero_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        self.data = public_train_data(rootdir)["input"]

        self.mean_, self.std_ = self._preprocess_data()

    def _preprocess_data(self):
        mean_ = {k: [] for k in self.metero_cols}
        std_ = {k: [] for k in self.metero_cols}

        for station in self.data.values():
            for col in self.metero_cols:
                mean_[col].append(station["data"][col].mean())
                std_[col].append(station["data"][col].std())

                station["data"][col] = default_fillnan_fn(station["data"][col])

        for col in self.metero_cols:
            mean_[col] = sum(mean_[col]) / len(mean_[col])
            std_[col] = sum(std_[col]) / len(std_[col])

        return mean_, std_

    def __len__(self):
        total_len = 35064

        return (total_len - self.inseq_len) // self.outseq_len

    def __getitem__(self, idx):
        in_start_idx = idx * self.outseq_len
        in_end_idx = in_start_idx + self.inseq_len
        out_start_idx = in_end_idx
        out_end_idx = out_start_idx + self.outseq_len

        inputs = []
        in_locs = []
        src_nexts = []
        time = None

        for station in self.data.values():
            df = station["data"].iloc[in_start_idx : in_end_idx]
            next_df = station["data"].iloc[out_start_idx : out_end_idx]

            metero = self._metero_to_tensor(df, usecols=self.metero_cols, norm=True)
            pm25_next = self._metero_to_tensor(next_df, usecols=["PM2.5"], norm=False).squeeze_(-1)

            if time is None:
                time = self._time_to_tensor(df)

            inputs.append(metero)
            in_locs.append(torch.tensor(station["loc"], dtype=torch.float))
            src_nexts.append(pm25_next)

        return {
            "metero": torch.stack(inputs, dim=0),
            "time": time,
            "src_nexts": torch.stack(src_nexts, dim=0),
            "src_locs": torch.stack(in_locs, dim=0)
        }

    def _metero_to_tensor(
        self,
        df: pd.DataFrame,
        usecols: List[str] = ["humidity", "temperature", "PM2.5"],
        norm: bool = True
    ):
        out = []

        for col in usecols:
            assert df[col].isna().sum() == 0, "Data must not contains nan values"
            datacol = df[col].to_numpy(dtype=float)
            datacol = torch.from_numpy(datacol)

            if norm:
                datacol = (datacol - self.mean_[col]) / self.std_[col]

            out.append(datacol)

        return torch.stack(out, dim=-1)

    def _time_to_tensor(
        self,
        df: pd.DataFrame,
    ):
        time = {
            "hour": df["hour"].tolist(),
            "day": df["day"].tolist(),
            "solar_term": [],
            "month": df["month"].tolist(),
            "year": df["year"].tolist(),
        }

        for i in range(len(df)):
            date = datetime(
                time["year"][i],
                time["month"][i],
                time["day"][i],
                time["hour"][i]
            )

            time["solar_term"].append(get_solar_term(date))

        for k in time:
            time[k] = torch.tensor(time[k], dtype=torch.long)

        return time
