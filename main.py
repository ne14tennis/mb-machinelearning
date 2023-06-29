import numpy as np
import pandas as pd
from aws_to_df import AwsToDf
import time
def run_program(name):
## Main df data
    atd = AwsToDf()
    df = atd.sql_to_df('test_sql')
    pd.set_option('display.max_columns', None)
    print(df.head())
## Series data df
    atd = AwsToDf()
    series = atd.sql_to_df('series_sql')
    pd.set_option('display.max_columns', None)
    print(series.head())
## Extracting day and time per show for most occurences ----split, find, replace
    df['air_date_time']=df['air_date_time'].str.strip()
    df[['date','time']]=df['air_date_time'].str.split(' ', 1, expand=True)
## coverting to datetime data type
    df['date']= pd.to_datetime(df['date'])
    df['time']=pd.to_datetime(df['time'])
##extracting day from datetime
    df['day']=df['date'].dt.day_name()
    df=df.drop(['date'], axis=1)
##rounding up time
    df['time'] = df['time'].apply(lambda x: x.round('30min'))
    df['time'] = df['time'].dt.strftime('%H:%M:%S')
    df=df.drop(['air_date_time'], axis=1)
    print(df.head())
    #lst_exc_view=['programme_duration','market_name','network_id',
     #           'station_id','series_name','series_id','is_latest_season',
      #          'genre_name','genre_id','is_national','market_id']

    ## creating df_series_1/2 by grouping for series-day/time,----------under_review
    ##   proportion of day per group, if proportion>80% we aggregate
    ## lst_day=[programme_duration','market_name','network_id','station_id','series_name','series_id','is_latest_season',
       ##      'day','genre_name','genre_id','is_national','market_id']
    ## df_series_1= df.groupby(lst_exc_view).size().reset_index(name='count')
    ## df_series_1.head()

    #creating season
    series['season']=series.groupby('name')['series_no'].rank()
    series=series.rename(columns={'name':'series_name', 'series_no':'series_id'})

#checking that shows have exclusive series_id
    c=series.groupby('series_name')['series_id'].apply(lambda x: x.duplicated().any().sum()).sum()
    series=series.drop(['series_name'], axis=1)
    print(c)
#c=0, therefore merge on only series_id is fine
    df=pd.merge(df,series,on='series_id',how='left')
    df.head()
##Summation per household season per season per station per day per time-slot, etc
    ##checking same series, different epsiode_id
    c1=df.groupby(['series_name','season','episode_id']).size().reset_index(name='count')
    ## multiple found, therefore summation on all other columns
    ### dropping
    new_df=df.drop(['episode_id','series_id','view_date_time','utc_airing_date'], axis=1)

    ## aggregating on is-national also(highest-freq/mode value as only 0.1% of observations very when grouped on given)
    new_df = new_df.groupby(['hh_id', 'day','time','market_name', 'network_id', 'station_id', 'series_name', 'season', 'is_latest_season', 'genre_name',
     'genre_id', 'market_id']).agg({'view_duration': 'sum', 'programme_duration': 'sum', 'is_national':lambda x:x.value_counts().index[0]}).reset_index()
    print(new_df.head())

## checking for duplicate combinations in each household
    duplicate_counts = new_df.duplicated(subset=['hh_id', 'series_name', 'season', 'network_id', 'station_id', 'day', 'time'], keep=False).groupby(new_df['hh_id']).sum()
    total_duplicates = duplicate_counts.sum()
    ## total_duplicates = 0, therefore we can proceed on these columns selection
    ## checking observations where view_duration>programee_duration
    df2=df[df['programme_duration']<df['view_duration']]
    ## as a percentage
    print("proportion="+ str(len(df2)/len(df)))

    ## creating 'watched' column
    new_df['ratio'] = new_df['view_duration'] / new_df['programme_duration']
    new_df['watched'] = (new_df['ratio'] >= 0.4).astype(int)
    new_df=new_df.drop(['ratio'], axis=1)
    new_df.head()

    print(len(new_df))
#creating mart_id, market_name, hh_id
    mrkt_hh=new_df[['hh_id','market_id','market_name']].copy()
#dropping hh char
    new_df=new_df.drop(['market_id','market_name','view_duration','programme_duration'], axis=1)
    print(len(new_df))
    new_df.head()

#selecting random 18,000 unique households
    nf_df = new_df.copy()
    uniq_hh = new_df['hh_id'].unique()
    np.random.shuffle(uniq_hh)
    sel_ids = uniq_hh[:18000] # first 18k
    new_df=new_df[new_df['hh_id'].isin(sel_ids)]
    len(new_df)
    new_df.head()

#adding 0/1 obs
    start_time = time.time()
    print("sess_check")

    ## Creating hh_lst
    hh_lst = new_df['hh_id'].unique().tolist()
    print(hh_lst)

    ## Creating combinations_df with a temporary identifier
    combinations_df = new_df[['day', 'time', 'network_id', 'station_id', 'series_name', 'season',
                          'is_latest_season', 'genre_name', 'genre_id', 'is_national']].drop_duplicates()
    combinations_df['temp_id'] = range(len(combinations_df))  # temporary identifier column
    print(len(combinations_df))

    section_start_time = time.time()

    merged_df = pd.merge(new_df, combinations_df, on=['day', 'time', 'network_id', 'station_id',
                                                  'series_name', 'season', 'is_latest_season',
                                                  'genre_name', 'genre_id', 'is_national'], how='left')

    for hh_id in hh_lst:
        iteration_start_time = time.time()

    ## Creating a copy
        temp_df = merged_df.copy()
        temp_df = temp_df[temp_df['hh_id'] == hh_id]  # Filter by household ID

    ## Getting the distinct list of temp_ids for that hh_id in temp_df
        distinct_temp_ids = temp_df['temp_id'].unique()

    ## Getting missing comb from combinations_df - comb of temp
        missing_comb = combinations_df[~combinations_df['temp_id'].isin(distinct_temp_ids)]

    ## Addding columns 'hh_id' and 'watched' to missing_comb
        missing_comb.loc[:, 'hh_id'] = hh_id
        missing_comb.loc[:, 'watched'] = 0

     ## Ordering columns acc new_df
        missing_comb = missing_comb[new_df.columns]

    # Concatenateing
        new_df = pd.concat([new_df, missing_comb], ignore_index=True)

        print(f"Household ID: {hh_id}")
        print(f"Number of Existing Combinations: {len(temp_df)}")
        print(f"Number of Missing Combinations: {len(missing_comb)}")
        print(f"Length of updated df: {len(new_df)}")
        iteration_end_time = time.time()
        iteration_elapsed_time = iteration_end_time - iteration_start_time
        print(f"Iteration Elapsed Time: {iteration_elapsed_time} seconds")
        print("---------------------------------")

    section_end_time = time.time()
    section_elapsed_time = section_end_time - section_start_time

    print(f"Elapsed Time: {section_elapsed_time} seconds")
    print(f"Total Number of Rows in new_df: {len(new_df)}")
    print(f"Total Number of Household IDs: {len(hh_lst)}")
    print(f"Total Number of Combinations: {len(combinations_df)}")

    total_elapsed_time = time.time() - start_time
    print(f"Total Elapsed Time: {total_elapsed_time} seconds")

## adding dem
    #extracting segment_id per hhh, some hh have multiple seg_ids so join with space
    #dem= df.groupby('hh_id')['segment_id'].apply(lambda x: ' '.join(map(str, x))).reset_index()
    #counting the max spaces(segments per hh) to create that many columns
    #

##checking if different networks have same show
    df3=df.groupby('series_name')['network_id'].nunique()
    df3.head()
    ## there are shows which are screened by multiple networks (list-netwotk_id+series)
    ## are they different shows or the same show screened by multiple networks?
    df3 = df.groupby('series_name')['network_id'].unique()

    df3.head()







if __name__ == '__main__':
    run_program('PyCharm')

