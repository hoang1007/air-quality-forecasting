from typing import Dict, List, Optional, Tuple
from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .raw_data import private_train_data, private_test_data
from utils.functional import extract_wind
from utils.export import batch_export


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
        split_ids = self._split_train_val(self.train_ratio)

        print(split_ids)

        self.data_train = PrivateDataset(
            self.rootdir,
            self.normalize_mean,
            self.normalize_std,
            split_ids=split_ids,
            data_set="train"
        )

        self.data_val = PrivateDataset(
            self.rootdir,
            self.normalize_mean,
            self.normalize_std,
            split_ids=split_ids,
            data_set="val"
        )

    def _split_train_val(self, train_ratio: float):
        total_len = 243
        train_size = int(total_len * train_ratio)

        perm_ids = torch.randperm(total_len, generator=torch.Generator().manual_seed(3107)) # consistent

        return perm_ids[:train_size], perm_ids[train_size:]

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
        split_ids: Tuple[torch.Tensor, torch.Tensor],
        data_set: str = "train",
    ):
        self.mean_ = normalize_mean
        self.std_ = normalize_std
        self.data_set = data_set
        self.inseq_len = 168
        self.outseq_len = 24

        self.num_stations = 71
        self.num_val_stations = 5
        self.num_train_stations = self.num_stations - self.num_val_stations

        if data_set in ("train", "val"):
            self.data = private_train_data(path.join(rootdir, "train"))
            self.train_ids, self.val_ids = split_ids
        elif data_set == "test":
            self.data = private_test_data(path.join(rootdir, "test"))

    def __len__(self):
        if self.data_set == "train":
            return self.train_ids.numel()
        elif self.data_set == "val":
            return self.val_ids.numel()
        else:
            return 89
    
    def __getitem__(self, idx):
        if self.data_set == "train":
            return self._get_training_item(self.train_ids[idx].item())
        elif self.data_set == "val":
            return self._get_training_item(self.val_ids[idx].item())
        else:
            return self._get_testing_item(idx)

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

        # thông tin của 71 trạm air và 143 trạm meteo
        in_air_ = torch.stack(in_air_, dim=0)
        in_meteo_ = torch.stack(in_meteo_, dim=0)
        air_locs_ = torch.tensor(air_locs_)
        meteo_locs_ = torch.tensor(meteo_locs_)
        targets_ = torch.stack(targets_, dim=0)

        # chia thành trạm nguồn và đích để train
        src_ids, tar_ids = self._split_stations()
        in_air_ = in_air_[src_ids]
        tar_locs_ = air_locs_[tar_ids]
        air_locs_ = air_locs_[src_ids]
        targets_ = targets_[tar_ids]

        return {
            "air": in_air_,
            "meteo": in_meteo_,
            "air_locs": air_locs_,
            "meteo_locs": meteo_locs_,
            "targets": targets_,
            "tar_locs": tar_locs_
        }

    def _get_testing_item(self, idx):
        item = self.data[idx]

        air_ = []
        meteo_ = []
        air_locs_ = []
        meteo_locs_ = []

        for station in item["air"].values():
            tmp_air = self._air_to_tensor(station["data"])
            air_locs_.append(station["loc"])
            air_.append(tmp_air)

        for station in item["meteo"].values():
            tmp_meteo = self._meteo_to_tensor(station["data"])
            meteo_locs_.append(station["loc"])
            meteo_.append(tmp_meteo)

        air_ = torch.stack(air_, dim=0)
        meteo_ = torch.stack(meteo_, dim=0)
        air_locs_ = torch.tensor(air_locs_)
        meteo_locs_ = torch.tensor(meteo_locs_)
        out_locs_ = torch.tensor(list(item["loc_output"].values()))

        return {
            "air": air_,
            "meteo": meteo_,
            "air_locs": air_locs_,
            "meteo_locs": meteo_locs_,
            "tar_locs": out_locs_,
            "folder_name": item["folder_name"]
        }

    def _split_stations(self):
        if self.data_set == "train":
            perm_ids = torch.randperm(self.num_train_stations)
            num_src_stations = self.num_train_stations - 10

            return perm_ids[:num_src_stations], perm_ids[num_src_stations:]
        elif self.data_set == "val":
            ids = torch.arange(self.num_train_stations + self.num_val_stations)

            return ids[:self.num_train_stations], ids[self.num_train_stations:]
        else:
            raise NotImplementedError
    
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
        evapo = torch.tensor(df["evaporation"].values, dtype=torch.float32)
        total_preci = torch.tensor(df["total_precipitation"].values, dtype=torch.float32)
      
        if norm:
            wind_spd = (wind_spd - self.mean_["wind_speed"]) / self.std_["wind_speed"]
            press = (press - self.mean_["surface_pressure"]) / self.std_["surface_pressure"]
            evapo = (evapo - self.mean_["evaporation"]) / self.std_["evaporation"]
            total_preci = (total_preci - self.mean_["total_precipitation"]) / self.std_["total_precipitation"]
        return torch.stack((wind_spd, press, evapo, total_preci), dim=-1)
        # return torch.stack((wind_spd, press), dim=-1)