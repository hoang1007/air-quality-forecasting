from os import path
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from .raw_data import air_quality_train_data, air_quality_test_data
from utils.functional import get_solar_term, get_next_period


class AirQualityDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        data_set: str = "train"
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
        time, time_next = None, None

        in_start_idx = idx * self.outseq_len
        in_end_idx = in_start_idx + self.inseq_len
        out_start_idx = in_end_idx
        out_end_idx = out_start_idx + self.outseq_len

        for station in self.data["input"].values():
            df = station["data"].iloc[in_start_idx : in_end_idx]

            metero = self._metero_to_tensor(df)
            if time is None:
                time, time_next = self._time_to_tensor(df)

            inputs.append(metero)
            in_locs.append(torch.tensor(station["loc"], dtype=torch.float))

        targets = []
        tar_locs = []
        for station in self.data["output"].values():
            df = station["data"].iloc[out_start_idx : out_end_idx]

            pm25 = self._metero_to_tensor(df, usecols=["PM2.5"]).squeeze_(-1)
            
            targets.append(pm25)
            tar_locs.append(torch.tensor(station["loc"], dtype=torch.float))

        return {
            "metero": torch.stack(inputs, dim=0),
            "time": time,
            "time_next": time_next,
            "src_locs": torch.stack(in_locs, dim=0),
            "targets": torch.stack(targets, dim=0),
            "tar_locs": torch.stack(tar_locs, dim=0),
        }

    def _get_testing_item(self, idx):
        dt = self.data["input"][idx]
        
        inputs = []
        in_locs = []
        time, time_next = None, None

        for station in dt.values():
            df = station["data"]

            metero = self._metero_to_tensor(df)
            inputs.append(metero)

            if time is None:
                time, time_next = self._time_to_tensor(df)

            in_locs.append(torch.tensor(station["loc"], dtype=torch.float))

        tar_locs = list(self.data["loc_output"].values())

        return {
            "metero": torch.stack(inputs, dim=0),
            "time": time,
            "time_next": time_next,
            "src_locs": torch.stack(in_locs, dim=0),
            "tar_locs": torch.tensor(tar_locs, dtype=torch.float),
            "folder_name": self.data["folder_name"][idx]
        }

    def _metero_to_tensor(
        self,
        df: pd.DataFrame,
        usecols: List[str] = ["humidity", "temperature", "PM2.5"]
    ):
        out = []

        for col in usecols:
            datacol = df[col].to_numpy(dtype=float)
            datacol = torch.from_numpy(datacol)
            datacol = (datacol - self.mean_[col]) / self.std_[col]

            out.append(datacol)

        return torch.stack(out, dim=-1)

    def _time_to_tensor(
        self,
        df: pd.DataFrame,
        return_time_next: bool = True
    ):
        time = {
            "hour": [],
            "day": [],
            "solar_term": []
        }

        for date in df["timestamp"]:
            date = datetime.strptime(date, "%d/%m/%Y %H:%M")

            time["hour"].append(date.hour)
            time["day"].append(date.day)
            time["solar_term"].append(get_solar_term(date))

        for k in time:
            time[k] = torch.tensor(time[k], dtype=torch.long)

        if return_time_next:
            time_next = {
                "hour": [],
                "day": [],
                "solar_term": []
            }

            for date in get_next_period(date, len=self.outseq_len):
                time_next["hour"].append(date.hour)
                time_next["day"].append(date.day)
                time_next["solar_term"].append(get_solar_term(date))

            for k in time_next:
                time_next[k] = torch.tensor(time_next[k], dtype=torch.long)
            
            return time, time_next
        else:
            return time