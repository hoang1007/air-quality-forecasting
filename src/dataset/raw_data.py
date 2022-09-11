from typing import Any, Dict, Tuple
from os import path, scandir
import pandas as pd


def private_train_data(rootdir: str):
    loc_air = _read_locations(path.join(rootdir, "air/location.csv"))
    loc_meteo = _read_locations(path.join(rootdir, "meteo/location.csv"))

    return {
        "air": _read_stations(path.join(rootdir, "air"), loc_air),
        "meteo": _read_stations(path.join(rootdir, "meteo"), loc_meteo)
    }

def private_test_data(rootdir: str):
    items = []
    for folder_name in scandir(rootdir):
        if path.isdir(folder_name):
            dirpath = folder_name.path

            loc_air = _read_locations(path.join(dirpath, "location_input.csv"))
            loc_meteo = _read_locations(path.join(dirpath, "meteo/location_meteorology.csv"))
            loc_output = _read_locations(path.join(dirpath, "location_output.csv"))
        
            items.append({
                "air": _read_stations(dirpath, loc_air),
                "meteo": _read_stations(path.join(dirpath, "meteo"), loc_meteo),
                "loc_output": loc_output,
                "folder_name": folder_name.name
            })
    
    return items

def public_train_data(rootdir: str):
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

def public_test_data(testdir: str, traindir: str):
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
            # usecols=["timestamp", "humidity", "temperature", "PM2.5"],
            index_col=False
        )

        stations[station_name] = {
            "loc": loc_map[station_name],
            "data": df
        }

    return stations

def _read_locations(filepath: str) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(filepath)

    try:
        return dict(zip(
            df["location"],
            zip(df["longitude"], df["latitude"])
        ))
    except: # meteorology format
        return dict(zip(
            df["stat_name"],
            zip(df["lon"], df["lat"])
        ))