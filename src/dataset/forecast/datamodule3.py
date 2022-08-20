from typing import Callable, List, Dict, Tuple, Optional
import os
from dataset.raw_data import *
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from datetime import datetime
from utils.functional import get_solar_term


def default_fillna_fn(x: pd.Series):
    return x.interpolate(option="spline").bfill()


class AirQualityDataModule3(LightningDataModule):
    def __init__(self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        split_stations: Tuple[int, int],
        fillnan_fn: Callable = default_fillna_fn,
        train_ratio: float = 0.75,
        batch_size: int = 32
    ):
        super().__init__()
        self.rootdir = rootdir
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.split_stations = split_stations
        self.fillnan_fn = fillnan_fn
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        datafull = AirQualityDataset3(
            self.rootdir,
            self.normalize_mean,
            self.normalize_std,
            self.split_stations,
            self.fillnan_fn,
            data_set="train",
        )

        train_size = int(len(datafull) * self.train_ratio)
        val_size = len(datafull) - train_size

        self.data_train, self.data_val = random_split(datafull, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=1)

class AirQualityDataset3(Dataset):
    def __init__(
        self,
        rootdir: str,
        normalize_mean: Dict[str, float],
        normalize_std: Dict[str, float],
        split_stations: Tuple[int, int],
        fillnan_fn: Callable = default_fillna_fn,
        data_set: str = "train",
        cachepath: str = None
    ):  
        self.in_stations, self.out_stations = split_stations
        assert self.in_stations + self.out_stations == 15

        self.data_set = data_set
        self.cachepath = cachepath
        self.inseq_len = 24 * 7
        self.outseq_len = 24
        self.feature_cols = ["humidity", "temperature", "PM2.5", "hour", "day", "month", "solar_term"]
        self.mean_ = normalize_mean
        self.std_ = normalize_std

        if cachepath is not None and os.path.isfile(cachepath):
            self.data, self._data_len = torch.load(cachepath)
        else:
            if data_set == "train":
                self.data, self._data_len = self._preprocess_training_data(
                    os.path.join(rootdir, "data-train"),
                    fillnan_fn,
                )
            elif data_set == "test":
                self.data, self._data_len = self._preprocess_testing_data(
                    os.path.join(rootdir, "public-test"),
                    os.path.join(rootdir, "data-train"),
                    fillnan_fn
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

    def save(self, path=None):
        if path is None:
            path = self.cachepath

        torch.save((self.data, len(self)), path)

    def _get_training_item(self, idx):
        # stations_dt: Tensor (n_stations, n_timesteps, n_features)
        # locs: Tensor (n_stations, 2)

        start_in_idx = idx * self.outseq_len
        end_in_idx = start_in_idx + self.inseq_len
        end_out_idx = end_in_idx + self.outseq_len

        ids = torch.randperm(self.in_stations + self.out_stations)
        in_ids = ids[:self.in_stations]
        out_ids = ids[self.in_stations:]

        pm25_idx = self.feature_cols.index("PM2.5")
        features = self.data["stations_dt"][in_ids, start_in_idx : end_in_idx]
        in_locs = self.data["locs"][in_ids]

        target = self.data["stations_dt"][out_ids, end_in_idx : end_out_idx, pm25_idx]
        tar_locs = self.data["locs"][out_ids]

        features = normalize_datatensor(features, self.feature_cols, self.mean_, self.std_)
        norm_target = normalize_datatensor(target.unsqueeze(-1), ["PM2.5"], self.mean_, self.std_).squeeze(-1)

        return {
            "features": features,
            "src_locs": in_locs,
            "tar_locs": tar_locs,
            "target": norm_target,
            "gt_target": target
        }

    def _get_testing_item(self, idx):
        dt = self.data["input"][idx]
        X_feats = normalize_datatensor(dt["stations_dt"], self.feature_cols, self.mean_, self.std_)

        return {
            "features": X_feats,
            "src_locs": dt["locs"],
            "tar_locs": self.data["y_locs"],
            "folder_idx": self.data["folder_idx"][idx],
        }

    def _preprocess_training_data(self, train_root: str, fillnan_fn: Callable):
        raw_data = air_quality_train_data(train_root)

        stations_dt = []
        locs = []

        for key in ("input", "output"):
            for station in raw_data[key].values():
                try:
                    # feat.shape == (n_timesteps, n_features)
                    # loc.shape == (2)
                    feat, loc = self._preprocess_station_data(station["data"], station["location"], fillnan_fn)

                    stations_dt.append(feat)
                    locs.append(loc)
                except:
                    continue

        stations_dt = torch.stack(stations_dt, dim=0) # (n_stations, n_timesteps, n_features)
        locs = torch.stack(locs, dim=0) # (n_stations, 2)

        data_len = (stations_dt.size(1) - self.inseq_len) // self.outseq_len

        return {
            "stations_dt": stations_dt,
            "locs": locs
        }, data_len

    def _preprocess_testing_data(self, test_root: str, train_root: str, fillnan_fn: Callable):
        raw_data = air_quality_test_data(test_root, train_root)

        items = []

        for item in raw_data["inputs"]:
            stations_dt = []
            locs = []
            for station in item.values():
                try:
                    feat, loc = self._preprocess_station_data(station["data"], station["location"], fillnan_fn)

                    stations_dt.append(feat)
                    locs.append(loc)
                except:
                    continue

            stations_dt = torch.stack(stations_dt, dim=0)
            locs = torch.stack(locs, dim=0)

            items.append({
                "stations_dt": stations_dt,
                "locs": locs
            })

        y_locs = []
        for loc in raw_data["output_location"].values():
            loc = torch.tensor(loc)
            y_locs.append(loc)

        y_locs = torch.stack(y_locs, dim=0)

        data_length = len(items)

        return {
            "input": items,
            "y_locs": y_locs,
            "folder_idx": raw_data["folder_idx"]
        }, data_length


    def _preprocess_station_data(self, df: pd.DataFrame, location: Tuple[float, float], fillnan_fn: Callable = None):
        if fillnan_fn is not None:
            for col in self.feature_cols:
                    if col in df.columns:
                        df[col] = fillnan_fn(df[col])

                        if df[col].isna().sum() > 0:
                            raise ValueError

        hours = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").hour)
        df["hour"] = hours / 24
        df["day"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").day / 31)
        df["month"] = df["timestamp"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").month / 12)
        df["solar_term"] = df["timestamp"].apply(lambda x: get_solar_term(datetime.strptime(x, "%d/%m/%Y %H:%M")) / 24)

        features = dataframe_to_tensor(df, usecols=self.feature_cols)
        loc = torch.tensor(location)

        return features, loc 


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