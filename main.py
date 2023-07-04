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

    series = atd.sql_to_df('series_sql')
    pd.set_option('display.max_columns', None)
    print(series.head())
## Demographics data df-1

    dem_df = atd.sql_to_df('dem_sql')
    pd.set_option('display.max_columns', None)
    print(dem_df.head())
## Demographics data df-2

    dem_nam = atd.sql_to_df('dem2_sql')
    pd.set_option('display.max_columns', None)
    print(dem_nam.head())
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

#Selecting random 16,000 unique households
    nf_df = new_df.copy()
    uniq_hh = new_df['hh_id'].unique()
    np.random.shuffle(uniq_hh)
    sel_ids = uniq_hh[:16000] # first 16k
    new_df=new_df[new_df['hh_id'].isin(sel_ids)]
    len(new_df)
    new_df.head()
## List of distinct 16k hh_ids chosen
    hh_lst= new_df['hh_id'].unique()
    hh_lst.head()
## Extracting dem_characteristics for sample
    dem_ref= hh_lst.merge(dem_df, on='hh_id', how='inner')
    ## Renaming and merging (note- obs lost on inner cause removing for unimportant segments----new_df should be filtered)
    dem_nam = dem_nam.rename(columns = {"id":"segment_id"})
    fin_dem = dem_nam.merge(dem_ref, on ='segment_id', how ='inner')
    ## To facilitate understanding
    fin_dem = fin_dem.groupby(['hh_id','segment_id','name'])
    fin_dem = fin_dem.first().reset_index()

    ## Creating dummies for segment_id
    dem_dum = pd.get_dummies(fin_dem.segment_id)

    ## Concatinating dummies to fin_dem['hh_id']
    segment_df = pd.concat([fin_dem['hh_id'], dem_dum], axis=1)
    ## Compressing dummies to hh_id level by summing
    segment_df = segment_df.groupby('hh_id').sum()
    segment_df = segment_df.reset_index()

    print(segment_df.head())
## Household and market_name
    #Filtering hh_id in mrkt_hh
    hh_lst = segment_df['hh_id'].unique()
    hh_df = pd.DataFrame(hh_lst, columns=['hh_id'])
    mrkt_hh = hh_df.merge(mrkt_hh, on= 'hh_id', how= 'inner').drop_duplicates()

    #Checking if any hh_id has more than 1 market_id
    multi_mrkt = (mrkt_hh.groupby('hh_id')['market_id'].size() > 1).any()
    #As False, dummy tranformation is left to later

#Filtering new_df acc hh_id in segment_df
    new_df = new_df[new_df['hh_id'].isin(hh_lst)]
    print(len(new_df))

#Adding 0/1 obs
    start_time = time.time()

    # Selecting first and last unique show observations and adding identifier column
    c_df = new_df.drop(['hh_id', 'watched'], axis=1)
    c1 = c_df.groupby('series_name').first().reset_index()
    c2 = c_df.groupby('series_name').last().reset_index()
    comb_df = pd.concat([c1, c2])
    comb_df = comb_df.drop_duplicates()
    print(len(comb_df))
    # As comb_df has only 1887 obs, we concatenate more examples (3500) from a sample of larger combs frame

    # Extracting sample
    all_combs = c_df.drop_duplicates()
    s_all_combs = all_combs.sample(n=3500, random_state=777)

    # Concatenating and dropping duplicates
    comb_df = pd.concat([comb_df, s_all_combs])
    comb_df = comb_df.drop_duplicates()
    print(len(comb_df))
    # Adding identifier
    comb_df['temp_id'] = range(len(comb_df))

    merged_df = pd.merge(new_df, comb_df, on=['day', 'time', 'network_id', 'station_id',
                                                  'series_name', 'season', 'is_latest_season',
                                                  'genre_name', 'genre_id', 'is_national'], how='left')
    merged_df.head()
    for hh_id in hh_lst:
        iteration_start_time = time.time()

        # Filter merged_df by household ID
        temp_df = merged_df[merged_df['hh_id'] == hh_id]

        # Getting the distinct list of temp_ids for that hh_id in temp_df
        distinct_temp_ids = temp_df['temp_id'].unique()

        # Getting missing comb from comb_df - combinations of temp
        missing_comb = comb_df[~comb_df['temp_id'].isin(distinct_temp_ids)]

        # Adding columns 'hh_id' and 'watched' to missing_comb
        missing_comb['hh_id'] = hh_id
        missing_comb['watched'] = 0

        # Ordering columns according to new_df
        missing_comb = missing_comb[new_df.columns]

        # Concatenating
        new_df = pd.concat([new_df, missing_comb], ignore_index=True)

        print(f"Household ID: {hh_id}")
        print(f"Number of Existing Combinations: {len(temp_df)}")
        print(f"Number of Missing Combinations: {len(missing_comb)}")
        print(f"Length of updated df: {len(new_df)}")
        iteration_end_time = time.time()
        iteration_elapsed_time = iteration_end_time - iteration_start_time
        print(f"Iteration Elapsed Time: {iteration_elapsed_time} seconds")
        print("---------------------------------")

    total_elapsed_time = time.time() - start_time
    print(f"Total Elapsed Time: {total_elapsed_time} seconds")
    print(f"Total Number of Rows in new_df: {len(new_df)}")
    print(f"Total Number of Household IDs: {len(hh_lst)}")
    print(f"Total Number of Combinations: {len(comb_df)}")

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

