import math
from typing import Dict, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from os import path
from dataset.raw_data import _read_stations, _read_locations
from utils.functional import euclidean_distance, haversine_distance


def imputation(rootdir: str, method: str = "idw", **kwargs):
    """
    Filling the missing data.

    Args:
        rootdir: str. Path to the data's directory.
        The directory's structure should contain a location csv file named `location.csv` or `location_input.csv`
        and multiple csv file denote the data of stations needed to impute.

    Note: Data in the csv files will be rewritten. Any sub-directory in the root directory will be ignored.
    """

    try:
        loc_map = _read_locations(path.join(rootdir, "location.csv"))
    except:
        loc_map = _read_locations(path.join(rootdir, "location_input.csv")) # test-set format

    data = _read_stations(rootdir, loc_map)

    print("Filling missing data")
    if method == "idw":
        data = idw_imputation(data, **kwargs)
    elif method == "spline":
        data = spline_imputation(data, **kwargs)
    elif method == "median":
        data = median_imputation(data, **kwargs)
    else:
        raise NotImplementedError

    print("Overwriting data to " + rootdir)
    _save(rootdir, data)


def spline_imputation(data: Dict):
    with tqdm(data.items(), colour="green") as bar:
        for name_i, station in bar:
            df = station["data"]
            assert isinstance(df, pd.DataFrame)

            # numeric_df = df.select_dtypes(include=np.number)
            # numeric_df.interpolate(method="spline", axis="columns", inplace=True)
            for col in df.columns:
                if df[col].dtype == np.number:
                    df[col].interpolate(option="spline", inplace=True)
                    df[col].bfill(inplace=True)
    
    return data

def median_imputation(data: Dict):
    with tqdm(data.items(), colour="green") as bar:
        for name_i, station in bar:
            df = station["data"]
            assert isinstance(df, pd.DataFrame)

            for col in df.columns:
                if df[col].dtype == np.number:
                    df[col].fillna(value=df[col].median(skipna=True), inplace=True)

    return data

def idw_imputation(data: Dict, dist_threshold: float = float("inf"), dist_type: str = "euclidean", beta: float = 1):
    with tqdm(data.items(), colour="green") as bar:
        for name_i, station in bar:
            df = station["data"]
            assert isinstance(df, pd.DataFrame)

            for col in df:
                # check missing values
                nan_ids = np.where(df[col].isna())[0]

                if nan_ids.size > 0: # if contains NaN values
                    collected = tuple(([], []) for _ in range(nan_ids.size)) # list of tuple (value, dist)
                    for name_j in data.keys():
                        if name_i != name_j:
                            if dist_type == 'haversine':
                                dist = haversine_distance(station["loc"], data[name_j]["loc"])
                            elif dist_type == 'euclidean':
                                dist = euclidean_distance(station["loc"], data[name_j]["loc"])
                            else:
                                raise NotImplementedError(f"Dont't have {dist_type} distance")

                            if dist <= dist_threshold:
                                temp_df = data[name_j]["data"]
                                assert isinstance(temp_df, pd.DataFrame)

                                # find possible values to impute
                                for i in range(nan_ids.size):
                                    val = temp_df[col].iloc[i]
                                    if not math.isnan(val):
                                        collected[i][0].append(val)
                                        collected[i][1].append(dist)

                    # impute nan values
                    for i in range(nan_ids.size):
                        values, dists = collected[i]
                        values = np.array(values)
                        dists = np.array(dists)

                        weights = np.power(dists, -beta)
                        weights = weights / weights.sum()

                        estimated_val = (values * weights).sum()

                        df.at[nan_ids[i], col] = estimated_val

    return data

def _save(rootdir: str, data: Dict):
    """
    Saves the data to the specified directory.
    :param rootdir: Directory to save the data.
    :param data: Data to save.
    :return:
    """
    for name, station in data.items():
        filepath = path.join(rootdir, name + ".csv")
        df = station["data"]
        assert isinstance(df, pd.DataFrame)
        df.to_csv(filepath, index=False)

if __name__ == '__main__':
    imputation("/home/hoang/Documents/CodeSpace/air-quality-forecasting/private-data/train/air", method="idw", dist_threshold=1000, beta=1.0)