import os
from typing import Dict, Tuple
import pandas as pd
from datetime import datetime

from sklearn.metrics import r2_score


def air_quality_train_data(root: str):
    '''
    Đọc dữ liệu `train` từ thư mục và trả về theo cấu trúc:
    >>> input: dict
    >>> ├─ station1: dict
    >>> │  ├─ location: tuple: (longitude, latitude)
    >>> │  ├─ timerange: tuple: (datetime, datetime)
    >>> │  ├─ humidity: list of float
    >>> │  ├─ temperature: list of float
    >>> │  ├─ pm2.5: list of float
    >>> ├─ station2: dict
    >>> ├─ .../
    >>> output: dict: giống như input
    '''
    location_map = _read_location_map(root)
    r2_score

    return {
        "input": _read_stations(os.path.join(root, "input"), location_map),
        "output": _read_stations(os.path.join(root, "output"), location_map),
    }

def air_quality_test_data(test_root: str, train_root: str):
    '''
    Đọc dữ liệu `test` từ thư mục và trả về theo cấu trúc:
    >>> input: list
    >>> ├─ 0/
    >>> │  ├─ station1: dict
    >>> │  │  ├─ location: tuple (longitude, latitude)
    >>> │  │  ├─ timerange: tuple (datetime, datetime)
    >>> │  │  ├─ humidity: list of float
    >>> │  │  ├─ temperature: list of float
    >>> │  │  ├─ pm2.5: list of float
    >>> │  ├─ station2: dict
    >>> │  ├─ .../
    >>> ├─ 1/
    >>> location_map/
    >>> ├─ station: dict {name: location}

    '''
    train_location_map = _read_location_map(train_root)

    data = {"input": [], "location_map": _read_location_map(test_root)}
    for dir in os.listdir(os.path.join(test_root, "input")):
        dir_path = os.path.join(test_root, "input", dir)
        if os.path.isdir(dir_path):
            data["input"].append(_read_stations(dir_path, train_location_map))
        else:
            raise ValueError("Invalid folder structure")
    
    return data    


def _read_stations(parent_dir, location_map: Dict[str, Tuple[float, float]]):
    stations = {}

    for file in os.listdir(parent_dir):
        filepath = os.path.join(parent_dir, file)
        if os.path.isfile(filepath):
            stations.update(_read_station_data(filepath, location_map))
        else:
            raise ValueError(
                "The parent directory should not contain any subdirectories. Dir " + filepath)

    return stations


def _read_station_data(file_path, location_map: Dict[str, Tuple[float, float]]) -> Dict:
    with open(file_path, "r") as f:
        station_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(
            f, usecols=["timestamp", "PM2.5", "humidity", "temperature"])

    start = datetime.strptime(df["timestamp"].iloc[0], "%d/%m/%Y %H:%M")
    end = datetime.strptime(df["timestamp"].iloc[-1], "%d/%m/%Y %H:%M")

    assert (end - start).total_seconds() // 3600 == len(df) - \
        1, "Timerange not matched"

    if station_name in location_map:
        location = location_map[station_name]
    else:
        location = None

    return {
        station_name: {
            "location": location,
            "timerange": (start, end),
            "pm2.5": df["PM2.5"].tolist(),
            "humidity": df["humidity"].tolist(),
            "temperature": df["temperature"].tolist(),
        }
    }


def _read_location_map(root):
    with open(os.path.join(root, "location.csv"), "r") as f:
        location_df = pd.read_csv(f, index_col=False)

    return dict(zip(
        location_df["station"],
        zip(location_df["longitude"], location_df["latitude"])
    ))

if __name__ == "__main__":
    train_data = air_quality_train_data("data/data-train")
    test_data = air_quality_test_data("data/public-test", "data/data-train")

    print(train_data.keys())
    print(train_data["input"].keys())

    print(test_data.keys())
    print(test_data["input"][0].keys())
    print(test_data["location_map"].keys())