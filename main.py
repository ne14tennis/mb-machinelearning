import numpy as np
import pandas as pd
from aws_to_df import AwsToDf
from newtools import PandasDoggo
import time
from sklearn.model_selection import StratifiedShuffleSplit
def run_program(name):
    # Loading refined dataframes
    atd = AwsToDf()
    new_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'ML_df3.csv','csv', has_header = True)
    segment_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'segment.csv','csv', has_header = True)
    mrkt_hh = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'market_hh.csv','csv', has_header = True)

    print('check 1')
    # Converting time to hour of day
    new_df['time'] = pd.to_datetime(new_df['time'])
    new_df['hour_of_day'] = new_df['time'].dt.hour
    new_df = new_df.drop(['time'], axis=1)
    print(new_df.head())
    # Merge
    n_df = new_df.merge(segment_df, on = "hh_id", how = 'left')
    n_df = n_df.merge(mrkt_hh, on = 'hh_id', how = 'left')
    n_df = n_df.drop(['market_name','series_name'], axis = 1)
    print(len(n_df))
    doggo = PandasDoggo()
    path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/Pd_df.csv"
    doggo.save(n_df, path)
    # Checking for NaN values
    n_df.isna().sum()

    # Getting dummies -----OHE
    day_dummy = pd.get_dummies(n_df.day)
    hh_dummy = pd.get_dummies(n_df.hh_id)
    network_dummy = pd.get_dummies(n_df.network_id)
    station_dummy = pd.get_dummies(n_df.station_id)
    season_dummy = pd.get_dummies(n_df.season)
    genre_dummy = pd.get_dummies(n_df.genre_id)
    mrkt_dummy = pd.get_dummies(n_df.market_id)

    print('Dummies created')















if __name__ == '__main__':
    run_program('PyCharm')

