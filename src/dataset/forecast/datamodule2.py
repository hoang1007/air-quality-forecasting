from typing import Callable, Dict, List, Tuple, Optional
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from dataset.raw_data import air_quality_train_data, air_quality_test_data
from utils.functional import get_solar_term


def default_fillna_fn(x: pd.Series):
    return x.interpolate(option="spline").bfill()


class AirQualityDataset2(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float] = None,
        normalize_std: Dict[str, float] = None,
        droprate: float = 0.5,
        fillnan_fn: Callable = None,
        fill_na: bool = True,
        data_set: str = "train",
        cachepath: str = None
    ):
        self.inframe_size = 24 * 7
        self.outframe_size = 24
        self.feature_cols = ["humidity", "temperature", "PM2.5"]
        self.time_cols = ["hour", "day", "month", "solar_term"]
        self.droprate = droprate
        self.data_set = data_set
        self.cachepath = cachepath
        self.fill_na = fill_na
        self.mean_, self.std_ = self._format_normalize(
            dict(normalize_mean), dict(normalize_std))

        if fillnan_fn is None and self.fill_na:
            fillnan_fn = default_fillna_fn

        if cachepath is not None and os.path.isfile(cachepath):
            self.data, self._data_len = torch.load(cachepath)
        else:
            if data_set == "train":
                self.data, self._data_len = self.preprocess_training_data(
                    os.path.join(rootdir, "data-train"),
                    fillnan_fn
                )
            elif data_set == "test":
                self.data, self._data_len = self.preprocess_testing_data(
                    os.path.join(rootdir, "public-test"),
                    os.path.join(rootdir, "data-train"),
                    fillnan_fn,
                )
            else:
                raise ValueError

    def save(self, path=None):
        if path is None:
            path = self.cachepath

        torch.save((self.data, len(self)), path)

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
        # X_feats: Tensor (data_len, n_stations, inframe_size, n_features)
        # X_locs: Tensor (n_stations, 2)
        # X_nexts: Tensor (data_len, n_stations, outframe_size)
        # X_masks: Tensor (data_len, n_stations)
        # y: Tensor (data_len, outframe_size)
        # y_locs: Tensor (data_len, 2)

        if index >= len(self):
            raise IndexError

        X_feats = normalize_datatensor(
            self.data["X_feats"][index], self.feature_cols, self.mean_, self.std_)
        X_feats = X_feats.nan_to_num_(nan=0)

        # X_nexts = normalize_datatensor(self.data["X_nexts"][index].unsqueeze(-1), ["PM2.5"], self.mean_, self.std_).squeeze(-1)
        # X_nexts = X_nexts.nan_to_num_(nan=0)
        X_nexts = self.data["X_nexts"][index]

        return {
            "features": X_feats,
            "src_locs": self.data["X_locs"],
            "src_nexts": X_nexts,
            "src_masks": self.data["X_masks"][index],
            "time": self.data["X_time"][index]
        }

    def _get_testing_item(self, index):
        # input: list of Dict
        # X_feats: Tensor (n_stations, n_timesteps, n_features)
        # X_locs: Tensor (n_stations, 2)
        # X_masks: Tensor (n_stations, n_frames)
        # y_locs: Tensor (n_target_stations, 2)

        dt = self.data["input"][index]
        X_feats = normalize_datatensor(
            dt["X_feats"], self.feature_cols, self.mean_, self.std_)
        X_feats = X_feats.nan_to_num_(nan=0)

        return {
            "features": X_feats,
            "src_locs": dt["X_locs"],
            "tar_locs": self.data["y_locs"],
            "src_masks": dt["X_masks"].squeeze(-1),
            "time": dt["X_time"],
            "folder_idx": self.data["folder_idx"][index],
        }

    def preprocess_training_data(self, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_train_data(train_root)
        pm25_idx = self.feature_cols.index("PM2.5")

        # X_feats.shape == (n_src_stations, n_timesteps, n_features)
        X_feats = torch.tensor([])
        X_locs = torch.tensor([])
        X_masks = torch.tensor([], dtype=torch.bool)
        X_time = torch.tensor([], dtype=torch.long)

        for stname, station in raw_data["input"].items():
            mask_ = None

            for col in ("humidity", "temperature", "PM2.5"):
                col_mask = self._create_mask(
                    station["data"][col], self.inframe_size, self.outframe_size)

                mask_ = col_mask if mask_ is None else torch.minimum(
                    mask_, col_mask)

            # mask_.shape == (n_frame,)
            # feat.shape == (9000, 3)
            # loc.shape == (2)
            feat, loc, time = self._preprocess_station_data(
                station["data"], station["location"], fillnan_fn)

            X_feats = torch.cat((X_feats, feat.unsqueeze(0)), dim=0)
            X_locs = torch.cat((X_locs, loc.unsqueeze(0)), dim=0)
            X_masks = torch.cat((X_masks, mask_.unsqueeze(0)), dim=0)
            X_time = torch.cat((X_time, time.unsqueeze(0)), dim=0)

        # convert X_feats, y from n_timesteps to n_frames
        __X_feats = []
        __X_nexts = []
        __X_time = []

        i = 0
        while True:
            start_idx = i * self.outframe_size
            end_idx = start_idx + self.inframe_size
            next_end_idx = end_idx + self.outframe_size

            if next_end_idx > X_feats.size(1):
                break
            __X_feats.append(X_feats[:, start_idx: end_idx].clone())
            __X_nexts.append(
                X_feats[:, end_idx: next_end_idx, pm25_idx].clone())
            __X_time.append(X_time[:, start_idx : end_idx].clone())

            i += 1

        # X_feats.shape == (n_src_stations, n_frames, inframe_size, n_features)
        # X_nexts.shape == (n_src_stations, n_frames, outframe_size)
        # X_masks.shape == (n_src_stations, n_frames)
        # y.shape == (n_target_stations, n_frames, outframe_size)
        # y_masks.shape == (n_target_stations, n_frames)

        return {
            "X_feats": __X_feats,
            "X_locs": X_locs,
            "X_masks": X_masks.t(),
            "X_nexts": __X_nexts,
            "X_time": __X_time
        }, len(__X_feats)

    def preprocess_testing_data(self, test_root: str, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_test_data(test_root, train_root)

        X_items = []
        for elm in raw_data["input"]:
            X_feats = torch.tensor([])
            X_locs = torch.tensor([])
            X_masks = torch.tensor([], dtype=torch.bool)
            X_time = torch.tensor([], dtype=torch.long)

            for stname, station in elm.items():
                mask_ = None
                for col in ("humidity", "temperature", "PM2.5"):
                    col_mask = self._create_mask(
                        station["data"][col], self.inframe_size, self.outframe_size)

                    mask_ = col_mask if mask_ is None else torch.minimum(
                        mask_, col_mask)

                feat, loc, time = self._preprocess_station_data(
                    station["data"], station["location"], fillnan_fn)

                X_feats = torch.cat((X_feats, feat.unsqueeze(0)), dim=0)
                X_locs = torch.cat((X_locs, loc.unsqueeze(0)), dim=0)
                X_masks = torch.cat((X_masks, mask_.unsqueeze(0)), dim=0)
                X_time = torch.cat((X_time, time.unsqueeze(0)), dim=0)

            X_items.append({
                "X_feats": X_feats,
                "X_locs": X_locs,
                "X_masks": X_masks,
                "X_time": X_time,
            })

        y_locs = torch.tensor([])

        for loc in raw_data["output_location"].values():
            loc = torch.tensor(loc)
            y_locs = torch.cat((y_locs, loc.unsqueeze(0)), dim=0)

        data_length = len(raw_data["input"])

        return {
            "input": X_items,
            "y_locs": y_locs,
            "folder_idx": raw_data["folder_idx"]
        }, data_length

    def _preprocess_station_data(self, df: pd.DataFrame, location: Tuple[float, float], fillnan_fn: Callable = None):
        time_ = {"hour": [], "day": [], "month": [], "solar_term": []}

        for t in df.timestamp:
            date = datetime.strptime(t, "%d/%m/%Y %H:%M")

            time_["hour"].append(date.hour)
            time_["day"].append(date.day)
            time_["month"].append(date.month)
            time_["solar_term"].append(get_solar_term(date))

        time = torch.tensor([], dtype=torch.long)
        for t in self.time_cols:
            time = torch.cat((time, torch.tensor(time_[t]).unsqueeze(-1)), dim=-1)

        if fillnan_fn is not None:
            for col in self.feature_cols:
                df[col] = fillnan_fn(df[col])

        features = dataframe_to_tensor(df, usecols=self.feature_cols)
        loc = torch.tensor(location)

        return features, loc, time

    def _format_normalize(self, mean_: Dict[str, float] = None, std_: Dict[str, float] = None):
        _autofill_cols = []

        if mean_ is None:
            mean_ = {feat: 0 for feat in self.feature_cols}
        else:
            for feat in self.feature_cols:
                if feat not in mean_:
                    mean_.update({feat: 0})
                    _autofill_cols.append(feat)

        if std_ is None:
            std_ = {feat: 1 for feat in self.feature_cols}
        else:
            for feat in self.feature_cols:
                if feat not in std_:
                    std_.update({feat: 1})

        if len(_autofill_cols):
            warnings.warn(f"{_autofill_cols} have filled with mean = 0 and std = 1")

        return mean_, std_

    def _create_mask(self, x: pd.Series, frame_size: int, stride: int):
        """
        Tạo mask cho từng frame

        Args:
            x: pandas.Series (n_timesteps,)

        Returns:
            mask: Tensor (n_frames,)
        """

        mask = []
        start_idx = 0

        while True:
            end_idx = start_idx + frame_size

            if end_idx > len(x):
                break

            nan_rate = x[start_idx: end_idx].isna().sum() / frame_size
            if nan_rate > self.droprate:
                mask.append(False)
            else:
                mask.append(True)

            start_idx += stride

        return torch.tensor(mask, dtype=torch.bool)


class AirQualityDataModule2(LightningDataModule):
    def __init__(self,
                 rootdir: str,
                 normalize_mean: Dict[str, float],
                 normalize_std: Dict[str, float],
                 droprate: float,
                 fillnan_fn: Callable = None,
                 fill_na: bool = True,
                 train_ratio: float = 0.75,
                 batch_size: int = 32,
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.rootdir = rootdir
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.train_ratio = train_ratio
        self.fillnan_fn = fillnan_fn
        self.fill_na = fill_na
        self.droprate = droprate

    def setup(self, stage: Optional[str] = None):
        datafull = AirQualityDataset2(
            self.rootdir,
            self.normalize_mean,
            self.normalize_std,
            self.droprate,
            data_set="train",
            fillnan_fn=self.fillnan_fn,
            fill_na=self.fill_na
        )

        train_size = int(len(datafull) * self.train_ratio)
        val_size = len(datafull) - train_size

        self.data_train, self.data_val = random_split(
            datafull, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=1)

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
        col_data = df[col].to_numpy()
        col_data = torch.from_numpy(col_data)

        features = torch.cat(
            (features, col_data.unsqueeze(-1)),
            dim=-1)

    return features