from random import sample
from typing import Callable, Dict, List, Tuple, Optional
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from dataset.raw_data import air_quality_train_data, air_quality_test_data


class AirQualityDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        droprate: float = 0.5,
        fillnan_fn: Callable = None,
        data_set: str = "train",
        cachepath: str = None
    ):
        self.inframe_size = 24 * 7
        self.outframe_size = 24
        self.feature_cols = ["humidity", "temperature", "PM2.5", "hour_sin", "hour_cos", "day", "month"]
        self.mean_ = normalize_mean
        self.std_ = normalize_std
        self.droprate = droprate
        self.fillnan_fn = fillnan_fn
        self.data_set = data_set

        if cachepath is not None and os.path.isfile(cachepath):
            self.data, self._data_len = torch.load(cachepath)
        else:
            if data_set == "train":
                self.data, self._data_len = self.preprocess_training_data(
                    os.path.join(rootdir, "data-train"),
                    fillnan_fn,
                )
            elif data_set == "test":
                self.data, self._data_len = self.preprocess_testing_data(
                    os.path.join(rootdir, "public-test"),
                    os.path.join(rootdir, "data-train"),
                    fillnan_fn,
                )
            else:
                raise ValueError

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        if self.data_set == "train":
            return self._get_training_item(index)
        elif self.data_set == "test":
            return self._get_testing_item(index)
        else:
            raise ValueError

    def _get_training_item(self, index):
        # X_feats: dict {station_name: feat (Tensor)} with feat.shape == (n_timesteps, n_features)
        # X_locs: dict {station_name: loc (Tensor)} with loc.shape == (2)
        # y: Tensor (n_target_stations, n_timesteps)
        # y_locs: Tensor (n_target_stations, 2)
        n_target_stations = self.data["y_locs"].size(0)
        n_timesteps = self.data["y"].size(1)

        tar_idx = index % n_target_stations
        sample_idx = index // n_target_stations

        src_start_idx = sample_idx * self.outframe_size
        src_end_idx = src_start_idx + self.inframe_size
        tar_start_idx = src_end_idx
        tar_end_idx = tar_start_idx + self.outframe_size

        if tar_end_idx > n_timesteps:
            raise IndexError
        
        X_feats, X_locs = self._get_features(self.data["X_feats"], self.data["X_locs"], src_start_idx, src_end_idx)
        gt_target = self.data["y"][tar_idx][tar_start_idx : tar_end_idx]
        norm_target = normalize_datatensor(gt_target.unsqueeze(-1), ["PM2.5"], self.mean_, self.std_).squeeze(-1)

        return {
            "features": X_feats,
            "src_locs": X_locs,
            "tar_loc": self.data["y_locs"][tar_idx],
            "target": norm_target,
            "gt_target": gt_target
        }

    def _get_testing_item(self, index):
        # input: list of Dict
            # X_feats: dict {station_name: feat (Tensor)} with feat.shape == (n_timesteps, n_features)
            # X_locs: dict {station_name: loc (Tensor)} with loc.shape == (2)
        # y_locs: Tensor (n_target_stations, 2)

        dt = self.data["input"][index]
        X_feats, X_locs = self._get_features(dt["X_feats"], dt["X_locs"], 0, None)

        return {
            "features": X_feats,
            "src_locs": X_locs,
            "tar_locs": self.data["y_locs"]
        }

    def _get_features(self,
        X_feats: Dict[str, torch.Tensor],
        X_locs: Dict[str, torch.Tensor],
        start_idx: int,
        end_idx: int
    ):
        feats = torch.tensor([])
        locs = torch.tensor([])

        # Kiểm tra nếu phần trăm missing data của trạm > droprate thì bỏ trạm đó
        for stname, all_feat in X_feats.items():
            # feat.shape == (n_samples, n_features)
            feat = all_feat[start_idx : end_idx].clone()
            nan_count = feat.isnan().sum(0)
            nan_rate = nan_count.max().item() / self.inframe_size

            if nan_rate < self.droprate:
                if self.fillnan_fn is not None:
                    for i in range(len(nan_count)):
                        if nan_count[i] > 0:
                            feat[:, i] = torch.tensor(self.fillnan_fn(pd.Series(feat[:, i])).to_numpy())

                feats = torch.cat((feats, feat.unsqueeze(0)), dim=0)
                locs = torch.cat((locs, X_locs[stname].unsqueeze(0)), dim=0)

        feats = normalize_datatensor(feats, self.feature_cols, self.mean_, self.std_)

        return feats, locs


    def preprocess_training_data(self, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_train_data(train_root)

        X_feats = {}
        X_locs = {}

        for stname, station in raw_data["input"].items():
            # feat.shape == (9000, 3)
            # loc.shape == (2)
            feat, loc = self._preprocess_station_data(
                station["data"], station["location"])

            X_feats[stname] = feat
            X_locs[stname] = loc

        y = torch.tensor([])
        y_locs = torch.tensor([])

        for station in raw_data["output"].values():
            feat, loc = self._preprocess_station_data(
                station["data"], station["location"], fillnan_fn)

            pm25_idx = self.feature_cols.index("PM2.5")
            y = torch.cat((y, feat[:, pm25_idx].unsqueeze(0)), dim=0)
            y_locs = torch.cat((y_locs, loc.unsqueeze(0)), dim=0)

        # data_length = n_target_stations * (num_timesteps - inframe_size) / outframe_size
        data_length = int((y.size(1) - self.inframe_size) / self.outframe_size) * y_locs.size(0)

        return {
            "X_feats": X_feats,
            "X_locs": X_locs,
            "y": y,
            "y_locs": y_locs,
        }, data_length

    def preprocess_testing_data(self, test_root: str, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_test_data(test_root, train_root)

        X_items = []
        for elm in raw_data["input"]:
            X_feats = {}
            X_locs = {}

            for stname, station in elm.items():
                feat, loc = self._preprocess_station_data(
                    station["data"], station["location"])

                X_feats[stname] = feat
                X_locs[stname] = loc

            X_items.append({
                "X_feats": X_feats,
                "X_locs": X_locs,
            })

        y_locs = torch.tensor([])

        for loc in raw_data["output_location"].values():
            loc = torch.tensor(loc)
            y_locs = torch.cat((y_locs, loc.unsqueeze(0)), dim=0)

        data_length = len(raw_data["input"])

        return {
            "input": X_items,
            "y_locs": y_locs
        }, data_length

    def _preprocess_station_data(self, df: pd.DataFrame, location: Tuple[float, float], fillnan_fn: Callable = None):
        hours = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").hour)
        df["hour_sin"] = np.sin(np.pi * hours / 12)
        df["hour_cos"] = np.cos(np.pi * hours / 12)

        df["day"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").day / 31)
        df["month"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").month / 12)

        if fillnan_fn is not None:
            for col in self.feature_cols:
                    df[col] = fillnan_fn(df[col])

        features = dataframe_to_tensor(df, usecols=self.feature_cols)
        loc = torch.tensor(location)

        return features, loc


class AirQualityDataModule(LightningDataModule):
    def __init__(self,
        rootdir: str,
        output_frame_size: int,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        stride: int = 1,
        fillnan_fn: Callable = None,
        train_ratio: float = 0.75,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.rootdir = rootdir
        self.output_frame_size = output_frame_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.stride = stride
        self.train_ratio = train_ratio
        self.fillnan_fn = fillnan_fn

    def setup(self, stage: Optional[str] = None):
        datafull = AirQualityDataset(
            self.rootdir,
            self.output_frame_size,
            self.normalize_mean,
            self.normalize_std,
            stride=self.stride,
            data_set="train",
            fillnan_fn=self.fillnan_fn
        )

        train_size = int(len(datafull) * self.train_ratio)
        val_size = len(datafull) - train_size

        self.data_train, self.data_val = random_split(datafull, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=1)

def normalize_datatensor(
    data: torch.Tensor,
    col_order: List[str],
    mean_: Dict[str, float],
    std_: Dict[str, float]
    ):
    """
    Normalize the data with `mean_` and `std_`

    Args:
        data: Tensor (...,n_features)
        col_order: List of column's names by order of features
        mean_: Dictionary of mean value for each column
        std_: Dictionary of std value for each column
    """

    data = data.clone()
    for i in range(data.size(-1)):
        feat = col_order[i]

        if feat in mean_:
            data[..., i] = (data[..., i] - mean_[feat]) / std_[feat]
    
    return data

def dataframe_to_tensor(df: pd.DataFrame, usecols: List[str]):
    """
    Convert dataframe to Tensor

    Args:
        df: pandas.DataFrame
        usecols: List of column names to convert
    """
    features = torch.tensor([])
    for col in usecols:
        features = torch.cat(
            (features, torch.tensor(df[col], dtype=torch.float).unsqueeze(-1)), 
        dim=-1)

    return features


if __name__ == "__main__":
    dts = AirQualityDataset(
        "./data",
        normalize_mean={"humidity": 0, "temperature": 0, "PM2.5": 0},
        normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        fillnan_fn=lambda x: x.interpolate(option="spline").bfill(),
        data_set="test"
    )

    print(dts[0])