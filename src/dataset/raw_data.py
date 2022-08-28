from typing import Any, Dict, Tuple
from os import path, scandir
import pandas as pd


def air_quality_train_data(rootdir: str):
    '''
    Dict dữ liệu gồm `input` và `output`.
    Dữ liệu các trạm được lưu dưới dạng:
        station_name:
            data: pandas.DataFrame
            loc: tuple (float, float)
    '''

    loc_input = _read_locations(path.join(rootdir, "location_input.csv"))
    loc_output = _read_locations(path.join(rootdir, "location_output.csv"))

    return {
        "input": _read_stations(path.join(rootdir, "input"), loc_input),
        "output": _read_stations(path.join(rootdir, "output"), loc_output),
    }

def air_quality_test_data(testdir: str, traindir: str):
    '''
    Dict dữ liệu gồm `input`, `loc_output`, `folder_name`.
    
    Dữ liệu các trạm được lưu dưới dạng:
        station_name:
            data: pandas.DataFrame
            loc: tuple (float, float)
    '''

    loc_input = _read_locations(path.join(traindir, "location_input.csv"))
    loc_output = _read_locations(path.join(testdir, "location.csv"))
    data = {"input": [], "loc_output": loc_output, "folder_name": []}

    root = path.join(testdir, "input")
    for folder_name in scandir(root):
        if path.isdir(folder_name):
            dirpath = path.join(root, folder_name)
            item = _read_stations(dirpath, loc_input)

            data["input"].append(item)
            data["folder_name"].append(folder_name)

    return data

def _read_stations(
    rootdir: str,
    loc_map: Dict[str, Tuple[float, float]]
):

    stations = {}

    for station_name in loc_map:
        filepath = path.join(rootdir, station_name + ".csv")
        df = pd.read_csv(
            filepath,
            usecols=["timestamp", "humidity", "temperature", "PM2.5"],
            index_col=False
        )

        stations[station_name] = {
            "loc": loc_map[station_name],
            "data": df
        }

    return stations

def _read_locations(filepath: str) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(filepath)

    return dict(zip(
        df["station"],
        zip(df["longitude"], df["latitude"])
    ))