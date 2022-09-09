from typing import Tuple
from unittest import TestCase
import torch
import pandas as pd
import math
from dataset.raw_data import public_train_data
from dataset import AirQualityDataset


class DataTest(TestCase):
    def test_contiguous(self, data: AirQualityDataset):
        raw = public_train_data("/home/hoang/Documents/CodeSpace/air-quality-forecasting/data/data-train")

        pm25_col_idx = data.feature_cols.index("PM2.5")
        target_stations = [
            "S0000328-Luong The Vinh",
            "S0000367-To Hieu",
            "S0000339-Kim Ma",
            "S0000182-Truong THCS Trung Hoa"
        ]

        src_stations = [
            "S0000153-Truong Tieu hoc Tran Quoc Toan",
            "S0000229-Quan Hoa",
            "S0000143-Thu vien - DHQG Ha Noi",
            "S0000310-Hang Trong",
            "S0000137-Ngoc Khanh",
            "S0000238-He thong lien cap Lomonoxop - Mam non",
            "S0000171-GENESIS School",
            "S0000210-Truong THCS Yen So",
            "S0000370-Ba Trieu",
            "S0000541-Tran Quang Khai",
            "S0000264-FDS - Ton That Thuyet"
        ]

        for i in range(len(data)):
            dt = data[i]

            tar_st = target_stations[dt["target_idx"]]
            frame_idx = dt["frame_idx"]
            src_start_idx = frame_idx * data.outframe_size
            src_end_idx = src_start_idx + data.inframe_size
            tar_end_idx = src_end_idx + data.outframe_size

            # check locations
            self.check_loc(dt["tar_loc"], raw["output"][tar_st]["location"])

            for j in range(dt["src_locs"].size(0)):
                self.check_loc(dt["src_locs"][j], raw["input"][src_stations[j]]["location"])

            # check data matching
            t = self.check_frame(dt["gt_target"], raw["output"][tar_st]["data"]["PM2.5"], src_end_idx, tar_end_idx)
            self.assertTrue(t, msg=f"Test faild at station {tar_st} in range ({src_end_idx}-{tar_end_idx})")

            for j in range(dt["src_locs"].size(0)):
                station_name = src_stations[j]
                t = self.check_frame(
                    dt["features"][j, :, pm25_col_idx],
                    raw["input"][station_name]["data"]["PM2.5"],
                    src_start_idx, src_end_idx
                )

                self.assertTrue(t, msg=f"Test faild at station {station_name} in range ({src_start_idx}-{src_end_idx})")

    def check_loc(self, x: torch.Tensor, y: Tuple[float, float]):
        self.assertTrue(torch.equal(x, torch.tensor(y)))
    
    def check_frame(
        self,
        col_data: torch.Tensor,
        raw_data: pd.Series,
        start_idx: int, end_idx: int
    ):
        EPS = 1e-4
        start_val = col_data[0].item()
        end_val = col_data[-1].item()

        gtsv = raw_data[start_idx]
        gtev = raw_data[end_idx - 1]

        if math.isnan(gtsv) or math.isnan(gtev):
            return True
        else:
            diff_s = abs(start_val - gtsv)
            diff_e = abs(end_val - gtev)
            
            return diff_s <= EPS and diff_e <= EPS

if __name__ == "__main__":
    train_dts = AirQualityDataset(
        "/home/hoang/Documents/CodeSpace/air-quality-forecasting/data",
        normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        droprate=1,
        data_set="train"
    )

    DataTest().test_contiguous(train_dts)