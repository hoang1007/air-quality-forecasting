from typing import Callable, Dict, List, Tuple, Optional
import os
from matplotlib.pyplot import fill
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
        output_frame_size: int,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        stride: int = 1,
        data_set: str = "train",
        fillnan_fn: Callable = None,
    ):
        self.input_frame_size = 168  # 24x7
        self.output_frame_size = output_frame_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.stride = stride
        self.data_set = data_set
        self.feature_cols = ["humidity", "temperature", "PM2.5"]

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
            raise ValueError(f"Unknown data set {self.data_set}")

    def __len__(self):
        return self._data_len
    
    def __getitem__(self, idx):
        if self.data_set == "train":
            return self._get_training_item(idx)
        elif self.data_set == "test":
            return self._get_testing_item(idx)
        else:
            raise ValueError(f"Unknown data set {self.data_set}")

    def preprocess_training_data(self, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_train_data(train_root)

        X_feats = torch.tensor([])
        X_loc = torch.tensor([])

        for station in raw_data["input"].values():
            feat, loc = self._preprocess_station_data(
                station["data"], station["location"], fillnan_fn)

            X_feats = torch.cat((X_feats, feat.unsqueeze(0)), dim=0)
            X_loc = torch.cat((X_loc, loc.unsqueeze(0)), dim=0)

        y_feats = torch.tensor([])
        y_loc = torch.tensor([])

        for station in raw_data["output"].values():
            try:
                feat, loc = self._preprocess_station_data(
                    station["data"], station["location"], fillnan_fn)

                pm25_idx = self.feature_cols.index("PM2.5")
                y_feats = torch.cat((y_feats, feat[:, pm25_idx].unsqueeze(0)), dim=0)
                y_loc = torch.cat((y_loc, loc.unsqueeze(0)), dim=0)
            except:
                continue

        # data_length = (num_timesteps - (input_frame_size + output_frame_size)) x n_target_stations
        data_length = (X_feats.size(1) - (self.input_frame_size +
                       self.output_frame_size)) * y_loc.size(0)

        return {
            "X_feats": X_feats,
            "X_locs": X_loc,
            "y_feats": y_feats,
            "y_locs": y_loc,
        }, data_length

    def _get_training_item(self, idx):
        n_target_stations = self.data["y_locs"].size(0)

        tar_station_dix = idx % n_target_stations
        sample_idx = (idx // n_target_stations) * self.stride

        src_end_idx = sample_idx + self.input_frame_size
        tar_end_idx = src_end_idx + self.output_frame_size

        # features.shape == (n_stations, input_frame_size, n_features)
        features = self.data["X_feats"][:, sample_idx: src_end_idx]
        features = self.normalize_datatensor(features, self.feature_cols)

        # gt_target.shape == (output_frame_size,)
        gt_target = self.data["y_feats"][tar_station_dix][src_end_idx: tar_end_idx]
        norm_target = self.normalize_datatensor(gt_target.unsqueeze(-1), ["PM2.5"]).squeeze(-1)

        return {
            "features": features,
            "src_locs": self.data["X_locs"],
            "tar_loc": self.data["y_locs"][tar_station_dix],
            "target": norm_target,
            "gt_target": gt_target
        }

    def preprocess_testing_data(self, test_root: str, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_test_data(test_root, train_root)

        X_items = []
        for elm in raw_data["input"]:
            X_feats = torch.tensor([])
            X_loc = torch.tensor([])

            for station in elm.values():
                try:
                    feat, loc = self._preprocess_station_data(
                        station["data"], station["location"], fillnan_fn)

                    X_feats = torch.cat((X_feats, feat.unsqueeze(0)), dim=0)
                    X_loc = torch.cat((X_loc, loc.unsqueeze(0)), dim=0)
                except:
                    continue

            X_items.append({
                "X_feats": X_feats,
                "X_locs": X_loc
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

    def _get_testing_item(self, idx):
        features = self.data["input"][idx]["X_feats"]
        features = self.normalize_datatensor(features, self.feature_cols)

        return {
            "features": features,
            "src_locs": self.data["input"][idx]["X_locs"],
            "tar_locs": self.data["y_locs"]
        }

    def _preprocess_station_data(self, df: pd.DataFrame, location: Tuple[float, float], fillnan_fn: Callable = None):
        # kiem tra neu du lieu trong thi drop station
        for f in self.feature_cols:
            if df[f].isna().sum() == len(df):
                raise Exception("Empty columns")

        if fillnan_fn is not None:
            for f in self.feature_cols:
                df[f] = fillnan_fn(df[f])

        features = self.dataframe_to_tensor(df, usecols=self.feature_cols)

        location = torch.tensor(location)

        return features, location

    def normalize_datatensor(self, data: torch.Tensor, col_order: List[str]):
        data = data.clone()
        # data.shape == (..., n_features)
        for i in range(data.size(-1)):
            feat = col_order[i]

            data[..., i] = (data[..., i] - self.normalize_mean[feat]) / self.normalize_std[feat]
        
        return data

    def normalize_dataframe(self, df: pd.DataFrame):
        for col in df:
            if df[col].dtype == float and col in self.normalize_mean:
                df[col] = (df[col] - self.normalize_mean[col]) / \
                    self.normalize_std[col]

        return df

    def dataframe_to_tensor(self, df: pd.DataFrame, usecols):
        features = torch.tensor([])
        for col in usecols:
            features = torch.cat(
                (features, torch.tensor(df[col], dtype=torch.float).unsqueeze(-1)), 
            dim=-1)

        return features


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
        self.train_ratio = train_ratio
        self.fillnan_fn = fillnan_fn

    def setup(self, stage: Optional[str] = None):
        datafull = AirQualityDataset(
            self.rootdir,
            self.output_frame_size,
            self.normalize_mean,
            self.normalize_std,
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


if __name__ == "__main__":
    dts = AirQualityDataset(
        "./data",
        output_frame_size=24,
        normalize_mean={"humidity": 0, "temperature": 0, "PM2.5": 0},
        normalize_std={"humidity": 0.4, "temperature": 0.4, "PM2.5": 0.4},
        data_set="test",
        fillnan_fn=lambda x: x.interpolate(option="spline").bfill()
    )

    print(dts[0])