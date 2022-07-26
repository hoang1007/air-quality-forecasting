from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd


def air_quality_train_data(rootdir: str):
    input_loc = _read_location_map(os.path.join(rootdir, 'location_input.csv'))
    output_loc = _read_location_map(os.path.join(rootdir, 'location_output.csv'))

    return {
        "input": _read_stations(os.path.join(rootdir, "input"), input_loc),
        "output": _read_stations(os.path.join(rootdir, "output"), output_loc),
    }

def air_quality_test_data(test_rootdir:str, train_rootdir: str):
    input_loc = _read_location_map(os.path.join(train_rootdir, 'location_input.csv'))
    output_loc = _read_location_map(os.path.join(test_rootdir, 'location.csv'))

    data = {"input": [], "output_location": output_loc}

    for dir in os.listdir(os.path.join(test_rootdir, 'input')):
        dir_path = os.path.join(test_rootdir, 'input', dir)

        if os.path.isdir(dir_path):
            data["input"].append(_read_stations(dir_path, input_loc))
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


def _read_station_data(filepath: str, location_map: Dict[str, Tuple[float, float]]):
    df = pd.read_csv(
        filepath, 
        usecols=["timestamp", "PM2.5", "humidity", "temperature"])
    station_name = os.path.splitext(os.path.basename(filepath))[0]

    if station_name not in location_map:
        raise ValueError(f"Station {station_name} does not exist")

    return {
        station_name: {
            "location": location_map[station_name],
            "data": df
        }
    }

def _read_location_map(loc_path: str):
    with open(loc_path, "r") as f:
        location_df = pd.read_csv(f, index_col=False)

    return dict(zip(
        location_df["station"],
        zip(location_df["longitude"], location_df["latitude"])
    ))