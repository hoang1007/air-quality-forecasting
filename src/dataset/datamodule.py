from cmath import isnan
from typing import Callable, Dict
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from dataset.raw_data import air_quality_train_data, air_quality_test_data


class AirQualityDataset(Dataset):
    def __init__(self, root: str, data_set: str = "train", fillnan_fn: Callable = None):
        self.encoder_len = 24 * 7
        self.predict_len = 24
        self.data_set = data_set

        if data_set == "train":
            self.data = self._handle_training_data(
                os.path.join(root, "data-train"),
                fillnan_fn
            )
        elif data_set == "test":
            self.data = self._handle_testing_data(
                os.path.join(root, "public-test"),
                os.path.join(root, "data-train"),
                fillnan_fn
            )
        else:
            raise ValueError("Data must be either train or test. Got " + data_set)

    def __len__(self):
        return self._len_data

    def __getitem__(self, index):
        if self.data_set == "train":
            return self._get_training_item(index)
        elif self.data_set == "test":
            return self._get_testing_item(index)
        else:
            raise ValueError()
    
    def _handle_training_data(self, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_train_data(train_root, fillnan_fn)
        
        data = {}

        for key in raw_data:
            feats = torch.tensor([])
            locations = torch.tensor([])

            for station in raw_data[key].items():
                feat, loc = self._station_data_totensor(station)

                feats = torch.cat((feats, feat), dim=0)
                locations = torch.cat((locations, loc), dim=0)
                    

            data[key] = {
                "features": feats,
                "locations": locations
            }
        
        assert data["input"]["features"].size(1) == data["output"]["features"].size(1), "n_samples of input and output must be the same"
        
        # data["input"].shape == (n_stations, n_samples, n_features)
        self._len_data = data["input"]["features"].size(1) - (self.encoder_len + self.predict_len)
        
        return data

    def _get_training_item(self, idx):
        src_end_idx = idx + self.encoder_len
        tar_end_idx = src_end_idx + self.predict_len

        assert tar_end_idx <= len(self), "Index out of bounds"
        
        return {
            "src_loc": self.data["input"]["locations"],
            "tar_loc": self.data["output"]["locations"],
            "features": self.data["input"]["features"][:, idx : src_end_idx],
            "target": self.data["output"]["features"][:, src_end_idx : tar_end_idx]
        }


    def _handle_testing_data(self, test_root: str, train_root: str, fillnan_fn: Callable = None):
        raw_data = air_quality_test_data(test_root, train_root, fillnan_fn)

        data = {"input": []}

        for stations in raw_data["input"]:
            feats = torch.tensor([])
            locations = torch.tensor([])

            for station in stations.items():
                feat, loc = self._station_data_totensor(station)

                feats = torch.cat((feats, feat), dim=0)
                locations = torch.cat((locations, loc), dim=0)

            data["input"].append({
                "feat": feats,
                "loc": locations
            })

        target_loc = torch.tensor([])
        for loc in raw_data["location_map"].values():
            target_loc = torch.cat((target_loc, torch.tensor(loc).unsqueeze(0)), dim=0)

        data["target_loc"] = target_loc

        self._len_data = len(data["input"])

        return data

    def _get_testing_item(self, idx):
        return {
            "features": self.data["input"][idx]["feat"],
            "src_loc": self.data["input"][idx]["loc"],
            "tar_loc": self.data["target_loc"]
        }
        
    def _station_data_totensor(self, station: Dict):
        station_name, station = station
        # location.shape == (1, 2)
        location = torch.tensor(station["location"]).unsqueeze(0)

        # xoa tram do neu khong du du lieu
        if pd.isna(station["humidity"]).sum() == len(station["humidity"])\
            or pd.isna(station["temperature"]).sum() == len(station["temperature"])\
            or pd.isna(station["pm2.5"]).sum() == len(station["pm2.5"]):

            print("drop station " + station_name)
            return torch.tensor([]), torch.tensor([])

        # features.shape == (1, n_samples, n_features)
        features = torch.tensor([
            station["humidity"],
            station["temperature"],
            station["pm2.5"]
        ]).t().unsqueeze(0)

        assert features.isnan().sum() == 0, "Features must not have NaN values at " + station_name

        return features, location


if __name__ == "__main__":
    dts = AirQualityDataset("data/", data_set="train", fillnan_fn=lambda x: x.interpolate(option="spline"))[0]
