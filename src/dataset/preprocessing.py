import pandas as pd
import os
import shutil
from datetime import datetime
import math
import numpy as np
import copy
from typing import Callable

class DataPreprocessing:
    def __init__(self, source_path : str, destination_path : str, folder : str, imputation_method : str = 'inversed_distance', processing_method : str = 'iqr'):
        # Time Column
        self.time_name = 'timestamp'
        self.date_format = '%d/%m/%Y %H:%M'

        # Read Data
        print('Reading Data . . .', end = ' ')

        self.data = self.read_data(source_path)
        self.loc_pd = None
        self.location = self.read_loc(source_path, self.data.keys())

        print('Done!')

        # Imputation
        print('Filling Missing Data . . .', end = ' ')

        self.imputation_feature = ['PM2.5', 'temperature', 'humidity']
        self.fill_all_option = True
        self.data = self.imputation(self.data, imputation_method, self.location)

        print('Done!')

        # # Handle Outliers
        # print('Handling Outliers . . .', end = ' ')
        # self.processing_feature = ['PM2.5', 'temperature', 'humidity']
        # self.process(self.data)

        # print('Done!')

        # Save Data
        print('Saving Data . . .', end = ' ')
        self.save(destination_path, folder, self.data)
        print('Done!')

    '''
        Reading data Function
    '''
    def read_data(self, dir_path):
        data = {}
        for name in os.listdir(dir_path):
            path = os.path.join(dir_path, name)
            if "location" in name or not os.path.isfile(path):
                continue
            
            data_pd = pd.read_csv(path)
            if self.time_name not in data_pd.columns:
                self.time_name = 'time'
                self.date_format = '%Y-%m-%d %H:%M:%S'
            
            if self.time_name in data_pd.columns:
                hour, hsin, hcos, day, month, year = self.extractTimestamps(data_pd[self.time_name], self.date_format)
            else:
                hour, hsin, hcos, day, month, year = self.extractTimestamps(data_pd['timestamp'], '%d/%m/%Y %H:%M')
            
            data.update({
                name[0: -4]:
                {
                'PM2.5': data_pd['PM2.5'],
                'temperature': data_pd['temperature'],
                'humidity': data_pd['humidity'],
                'hour': hour,
                'hsin': hsin,
                'hcos': hcos,
                'day': day,
                'month': month,
                'year' : year
                }
            })
        return data

    def read_loc(self, dir_path, used_loc = []):
        loc_path = os.path.join(dir_path, 'location.csv')
        if not os.path.exists(loc_path):
            loc_path = os.path.join(dir_path, 'location_input.csv')
        locs_pd = pd.read_csv(loc_path)
        self.loc_pd = locs_pd
        locs_dict, longitude, latitude = {}, [], []

        main_name = 'location'
        if 'station' in locs_pd.columns:
                main_name = 'station'

        for i in range(len(locs_pd)):
            locs_dict.update({locs_pd[main_name][i] : [locs_pd['latitude'][i], locs_pd['longitude'][i]]})

        if not used_loc: used_loc = list(locs_pd[main_name])

        res = {}
        for loc in used_loc:
            res.update({loc : locs_dict[loc]})

        return res

    def extractDate(self, timestamp, date_format):
        tmp = datetime.strptime(timestamp, date_format)
        return tmp.hour, tmp.day, tmp.month, tmp.year

    def extractTimestamps(self, timestamps, date_format):
        hour, hsin, hcos, day, month, year = [], [], [], [], [], []
        for timestamp in timestamps:
            h, d, m, y = self.extractDate(timestamp, date_format)
            hour.append(h)
            hsin.append(math.sin(2*math.pi*h/24))
            hcos.append(math.cos(2*math.pi*h/24))
            day.append(d)
            month.append(m)
            year.append(y)
        return hour, hsin, hcos, day, month, year

    '''
        Imputation Function
    '''

    def imputation(self, data : dict, method : str = 'interpolate', location : dict = None):
        raw = copy.deepcopy(data)
        for station in data.keys():
            for feature in data[station].keys():
                if feature in self.imputation_feature:
                    if method == 'mean':
                        data[station][feature] = data[station][feature].fillna(value = data[station][feature].mean())

                    if method == 'median':
                        data[station][feature] = data[station][feature].fillna(value = data[station][feature].median())

                    if method == 'locf':
                        data[station][feature] = data[station][feature].fillna(method = 'bfill')

                    if method == 'nocf':
                        data[station][feature] = data[station][feature].fillna(method = 'ffill')

                    if method == 'linear':
                        data[station][feature] = data[station][feature].interpolate(option = 'linear').bfill()

                    if method == 'spline':
                        data[station][feature] = data[station][feature].interpolate(option = 'spline').bfill()

                    if method == 'time':
                        data[station][feature] = data[station][feature].interpolate(option = 'time').bfill()

                    if method == 'simple_moving_average':
                        while data[station][feature].isnull().any():
                            data[station][feature] = data[station][feature].rolling(min_periods = 1, center = True, window = 24).mean()

                    if method == 'inversed_distance':
                        data[station][feature] = self.inversed_distance_fill(raw, station, feature, location)
                        if data[station][feature].isnull().any():
                            data[station][feature] = data[station][feature].interpolate(option = 'spline').bfill()
        return data

    def distance(self, start : list, end : list, method : str = 'euclidean'):
        dist = 0
        for index in range(len(start)):
            if method == 'euclidean':
                dist += math.pow(start[index] - end[index], 2)
            if method == 'manhattan':
                dist += abs(start[index] - end[index])
        if method == 'euclidean':
            dist = math.sqrt(dist)
        return dist

    def check_na_feature(self, index : int, data : dict, station : str, feature : str):
        cnt = 0
        for check_feature in self.imputation_feature:
            if pd.isna(data[station][check_feature][index]):
                cnt += 1
        if self.fill_all_option:
            return pd.isna(data[station][feature][index])
        return cnt == len(self.imputation_feature)

    def inversed_distance_fill(self, data : dict, station : str, feature : str, location : list):
        new_data = data[station][feature].copy()
        for index in range(new_data.shape[0]):
            if self.check_na_feature(index, data, station, feature):
                new_data[index] = 0.0
                id = 0.0
                for other in location.keys():
                    if other != station:
                        if not pd.isna(data[other][feature][index]):
                            dist = self.distance(location[station], location[other])
                            id += 1/dist
                            new_data[index] += 1/dist * data[other][feature][index]
                if (id != 0):
                    new_data[index] /= id
                else:
                    new_data[index] = data[station][feature][index]
        return new_data

    '''
        Handle Outliers Function
    '''

    def process(self, data : dict, method : str = 'iqr'):
        for station in data.keys():
            for feature in self.processing_feature:
                self.cap_outliers(data[station][feature], threshold = 1.5, method = 'iqr')

    def cap_outliers(self, data : pd.DataFrame, threshold : float = 1.5, method : str = 'iqr'):
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lbound = Q1 - threshold*IQR
            ubound = Q3 + threshold*IQR

            outliers = (data < lbound) | (data > ubound)
            print('=', data[outliers], '\n')

            data = data.copy()
            data.loc[data < lbound] = data.loc[~outliers].min()
            data.loc[data > ubound] = data.loc[~outliers].max()

    '''
        Save Data Function
    '''

    def compressDate(self, hour : int, day : int, month : int, year : int, date_format):
        tmp = datetime(year = year, month = month, day = day, hour = hour, minute = 0, second = 0)
        return tmp.strftime(date_format)

    def save(self, parent_dir : str = 'data\\data-train-full\\', folder : str = "", data : dict = None):
        path = os.path.join(parent_dir, folder)
        # if os.path.exists(path):
        #     shutil.rmtree(path)
        # os.mkdir(path)

        self.loc_pd.to_csv(os.path.join(path, 'location.csv'))

        for station in data.keys():
            timesteps = []
            for index in range(len(data[station]['PM2.5'])):
                timesteps.append(self.compressDate(data[station]['hour'][index], data[station]['day'][index], data[station]['month'][index], data[station]['year'][index], self.date_format))

            dt = {
                self.time_name : timesteps,
                'PM2.5' : data[station]['PM2.5'].to_list(),
                'humidity' : data[station]['humidity'].to_list(),
                'temperature' : data[station]['temperature'].to_list()
            }
            df = pd.DataFrame(dt)

            save_path = os.path.join(path, station + '.csv')

            if os.path.exists(save_path):
                os.remove(save_path)
            df.to_csv(save_path)

if __name__ == '__main__':
    # # impute train data
    # data = DataPreprocessing(source_path = 'data-private/air', destination_path = 'data-private/', folder = 'air-full')

    # impute test data
    root_test = "data-private/test"
    for folder_name in os.scandir(root_test):
        # pathdir = os.path.join(root_test, folder_name)

        DataPreprocessing(source_path = folder_name.path, destination_path = root_test, folder = folder_name.name)