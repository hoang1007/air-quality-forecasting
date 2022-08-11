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
        self.feature_cols = ["humidity", "temperature", "PM2.5", "hour", "day", "month", "solar_term"]
        self.mean_ = normalize_mean
        self.std_ = normalize_std
        self.droprate = droprate
        self.fillnan_fn = fillnan_fn
        self.data_set = data_set

        if cachepath is not None and os.path.isfile(cachepath):
            self.data, self._data_len = torch.load(cachepath)
        else:
            if data_set in ("train", "val", "trainval"):
                self.data, self._data_len = self.preprocess_training_data(
                    os.path.join(rootdir, "data-train"),
                    data_set,
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

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        if self.data_set in ("train", "val", "trainval"):
            return self._get_training_item(index)
        elif self.data_set == "test":
            return self._get_testing_item(index)
        else:
            raise ValueError

    def _get_training_item(self, index):
        # X_feats: Tensor (n_stations, data_len, inframe_size, n_features)
        # X_locs: Tensor (n_stations, 2)
        # X_masks: Tensor (n_stations, data_len)
        # y: Tensor (data_len, outframe_size)
        # y_locs: Tensor (data_len, 2)

        if index >= len(self):
            raise IndexError
        
        X_feats = normalize_datatensor(self.data["X_feats"][:, index], self.feature_cols, self.mean_, self.std_)
        X_feats.nan_to_num_(nan=0)

        gt_target = self.data["y"][index]
        norm_target = normalize_datatensor(gt_target.unsqueeze(-1), ["PM2.5"], self.mean_, self.std_).squeeze(-1)

        return {
            "features": X_feats,
            "src_locs": self.data["X_locs"],
            "src_masks": self.data["X_masks"][:, index],
            "tar_loc": self.data["y_locs"][index],
            "target": norm_target,
            "gt_target": gt_target,
            "frame_idx": self.data["frame_ids"][index],
            "target_idx": self.data["target_ids"][index]
        }

    def _get_testing_item(self, index):
        # input: list of Dict
            # X_feats: Tensor (n_stations, n_timesteps, n_features)
            # X_locs: Tensor (n_stations, 2)
            # X_masks: Tensor (n_stations, n_frames)
        # y_locs: Tensor (n_target_stations, 2)

        dt = self.data["input"][index]
        X_feats = normalize_datatensor(dt["X_feats"], self.feature_cols, self.mean_, self.std_)
        X_feats.nan_to_num_(nan=0)

        return {
            "features": X_feats,
            "src_locs": dt["X_locs"],
            "src_masks": dt["X_masks"].squeeze(-1),
            "tar_locs": self.data["y_locs"],
            "folder_idx": self.data["folder_idx"][index],
        }


    def preprocess_training_data(self, train_root: str, data_set: str, fillnan_fn: Callable = None):
        raw_data = air_quality_train_data(train_root)

        if data_set == "train":
            target_ids = [0, 1]
        elif data_set == "val":
            target_ids = [2, 3]
        elif data_set == "trainval":
            target_ids = [0, 1, 2, 3]

        # X_feats.shape == (n_src_stations, n_timesteps, n_features)
        X_feats = torch.tensor([])
        X_locs = torch.tensor([])
        X_masks = torch.tensor([], dtype=torch.bool)

        for stname, station in raw_data["input"].items():
            mask_ = None

            for col in ("humidity", "temperature", "PM2.5"):
                col_mask = self._create_mask(station["data"][col], self.inframe_size, self.outframe_size)

                mask_ = col_mask if mask_ is None else torch.minimum(mask_, col_mask)

            # mask_.shape == (n_frame,)
            # feat.shape == (9000, 3)
            # loc.shape == (2)
            feat, loc = self._preprocess_station_data(
                station["data"], station["location"], fillnan_fn)

            X_feats = torch.cat((X_feats, feat.unsqueeze(0)), dim=0)
            X_locs = torch.cat((X_locs, loc.unsqueeze(0)), dim=0)
            X_masks = torch.cat((X_masks, mask_.unsqueeze(0)), dim=0)

        # y.shape == (n_target_stations, n_timesteps)
        y = torch.tensor([])
        y_locs = torch.tensor([])
        y_masks = torch.tensor([])

        for station in raw_data["output"].values():
            mask_ = self._create_mask(station["data"]["PM2.5"][self.inframe_size:], self.outframe_size, self.outframe_size)
            feat, loc = self._preprocess_station_data(
                station["data"], station["location"], fillnan_fn)

            pm25_idx = self.feature_cols.index("PM2.5")
            y = torch.cat((y, feat[:, pm25_idx].unsqueeze(0)), dim=0)
            y_locs = torch.cat((y_locs, loc.unsqueeze(0)), dim=0)
            y_masks = torch.cat((y_masks, mask_.unsqueeze(0)), dim=0)

        n_frames = y_masks.size(1)
        # convert X_feats, y from n_timesteps to n_frames
        __X_feats = torch.zeros((X_feats.size(0), n_frames, self.inframe_size, X_feats.size(-1)))

        for i in range(n_frames):
            start_idx = i * self.outframe_size
            end_idx = start_idx + self.inframe_size

            __X_feats[:, i] = X_feats[:, start_idx : end_idx].clone()

        # X_feats.shape == (n_src_stations, n_frames, inframe_size, n_features)
        # X_masks.shape == (n_src_stations, n_frames)
        # y.shape == (n_target_stations, n_frames, outframe_size)
        # y_masks.shape == (n_target_stations, n_frames)
        y = y[:, self.inframe_size:].reshape(-1, n_frames, self.outframe_size)
        X_feats = __X_feats
        X_masks = X_masks[:, :n_frames]

        __X_feats = []
        __X_masks = []
        __y = []
        __y_locs = []
        frame_ids = []
        target_station_ids = []

        n_target_stations = y.size(0)
        for i in range(n_frames):
            for j in range(n_target_stations):
                if j in target_ids and y_masks[j, i] == 1:
                    __X_feats.append(X_feats[:, i])
                    __X_masks.append(X_masks[:, i])
                    __y.append(y[j, i])
                    __y_locs.append(y_locs[j])
                    frame_ids.append(i)
                    target_station_ids.append(j)

        data_length = len(__y)

        __X_feats = torch.stack(__X_feats, dim=1)
        __X_masks = torch.stack(__X_masks, dim=1)
        __y = torch.stack(__y, dim=0)
        __y_locs = torch.stack(__y_locs, dim=0)

        return {
            "X_feats": __X_feats,
            "X_locs": X_locs,
            "X_masks": __X_masks,
            "y": __y,
            "y_locs": __y_locs,
            "frame_ids": frame_ids,
            "target_ids": target_station_ids
        }, data_length

    def preprocess_testing_data(self, test_root: str, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_test_data(test_root, train_root)

        X_items = []
        for elm in raw_data["input"]:
            X_feats = torch.tensor([])
            X_locs = torch.tensor([])
            X_masks = torch.tensor([], dtype=torch.bool)

            for stname, station in elm.items():
                mask_ = None
                for col in ("humidity", "temperature", "PM2.5"):
                    col_mask = self._create_mask(station["data"][col], self.inframe_size, self.outframe_size)

                    mask_ = col_mask if mask_ is None else torch.minimum(mask_, col_mask)

                feat, loc = self._preprocess_station_data(
                    station["data"], station["location"], fillnan_fn)

                X_feats = torch.cat((X_feats, feat.unsqueeze(0)), dim=0)
                X_locs = torch.cat((X_locs, loc.unsqueeze(0)), dim=0)
                X_masks = torch.cat((X_masks, mask_.unsqueeze(0)), dim=0)

            X_items.append({
                "X_feats": X_feats,
                "X_locs": X_locs,
                "X_masks": X_masks,
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

    def _get_solar_term(self, date: datetime):
        def in_range(date: datetime, d1: Tuple[int, int], d2: Tuple[int, int]):
            d1, m1 = d1
            d2, m2 = d2

            cond1 = date.day >= d1 and date.month == m1
            cond2 = date.month > m1
            cond3 = date.month < m2
            cond4 = date.day < d2 and date.month == m2
            
            return (cond1 or cond2) and (cond3 or cond4)

        LAP_XUAN, VU_THUY, KINH_TRAP, XUAN_PHAN, THANH_MINH, COC_VU,\
        LAP_HA, TIEU_MAN, MANG_CHUNG, HA_CHI, TIEU_THU, DAI_THU,\
        LAP_THU, XU_THU, BACH_LO, THU_PHAN, HAN_LO, SUONG_GIANG,\
        LAP_DONG, TIEU_TUYET, DAI_TUYET, DONG_CHI, TIEU_HAN, DAI_HAN,\
        = range(24)

        if in_range(date, (4, 2), (18, 2)):
            return LAP_XUAN
        elif in_range(date, (18, 2), (5, 3)):
            return VU_THUY
        elif in_range(date, (5, 3), (20, 3)):
            return KINH_TRAP
        elif in_range(date, (20, 3), (4, 4)):
            return XUAN_PHAN
        elif in_range(date, (4, 4), (20, 4)):
            return THANH_MINH
        elif in_range(date, (20, 4), (5, 5)):
            return COC_VU
        elif in_range(date, (5, 5), (21, 5)):
            return LAP_HA
        elif in_range(date, (21, 5), (5, 6)):
            return TIEU_MAN
        elif in_range(date, (5, 6), (21, 6)):
            return MANG_CHUNG
        elif in_range(date, (21, 6), (7, 7)):
            return HA_CHI
        elif in_range(date, (7, 7), (22, 7)):
            return TIEU_THU
        elif in_range(date, (22, 7), (7, 8)):
            return DAI_THU
        elif in_range(date, (7, 8), (23, 8)):
            return LAP_THU
        elif in_range(date, (23, 8), (7, 9)):
            return XU_THU
        elif in_range(date, (7, 9), (23, 9)):
            return BACH_LO
        elif in_range(date, (23, 9), (8, 10)):
            return THU_PHAN
        elif in_range(date, (8, 10), (23, 10)):
            return HAN_LO
        elif in_range(date, (23, 10), (7, 11)):
            return SUONG_GIANG
        elif in_range(date, (7, 11), (22, 11)):
            return LAP_DONG
        elif in_range(date, (22, 11), (7, 12)):
            return TIEU_TUYET
        elif in_range(date, (7, 12), (21, 12)):
            return DAI_TUYET
        elif in_range(date, (21, 12), (32, 12)) or in_range(date, (1, 1), (5, 1)):
            return DONG_CHI
        elif in_range(date, (5, 1), (20, 1)):
            return TIEU_HAN
        elif in_range(date, (20, 1), (4, 2)):
            return DAI_HAN
        else:
            raise ValueError

    def _preprocess_station_data(self, df: pd.DataFrame, location: Tuple[float, float], fillnan_fn: Callable = None):
        hours = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").hour)
        df["hour"] = hours / 24
        df["day"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").day / 31)
        df["month"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").month / 12)
        df["solar_term"] = df["timestamp"].apply(lambda x: self._get_solar_term(datetime.strptime(x, "%d/%m/%Y %H:%M")) / 24)

        if fillnan_fn is not None:
            for col in self.feature_cols:
                    df[col] = fillnan_fn(df[col])

        features = dataframe_to_tensor(df, usecols=self.feature_cols)
        loc = torch.tensor(location)

        return features, loc

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

            nan_rate = x[start_idx : end_idx].isna().sum() / frame_size
            if nan_rate > self.droprate:
                mask.append(False)
            else:
                mask.append(True)

            start_idx += stride

        return torch.tensor(mask, dtype=torch.bool)


class AirQualityDataModule(LightningDataModule):
    def __init__(self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        droprate: float,
        fillnan_fn: Callable = None,
        train_ratio: float = 0.75,
        split_mode="station",
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.rootdir = rootdir
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.train_ratio = train_ratio
        self.fillnan_fn = fillnan_fn
        self.droprate = droprate
        self.split_mode = split_mode

    def setup(self, stage: Optional[str] = None):
        if self.split_mode == "timestamp":
            datafull = AirQualityDataset(
                self.rootdir,
                self.normalize_mean,
                self.normalize_std,
                self.droprate,
                data_set="trainval",
                fillnan_fn=self.fillnan_fn
            )

            train_size = int(len(datafull) * self.train_ratio)
            val_size = len(datafull) - train_size

            self.data_train, self.data_val = random_split(datafull, [train_size, val_size])
        elif self.split_mode == "station":
            self.data_train = AirQualityDataset(
                self.rootdir,
                self.normalize_mean,
                self.normalize_std,
                self.droprate,
                data_set="train",
                fillnan_fn=self.fillnan_fn
            )

            self.data_val = AirQualityDataset(
                self.rootdir,
                self.normalize_mean,
                self.normalize_std,
                self.droprate,
                data_set="val",
                fillnan_fn=self.fillnan_fn
            )
        else:
            raise ValueError

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
        data_set="train"
    )

    print(dts[0])