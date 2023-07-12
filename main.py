import numpy as np
import pandas as pd
from aws_to_df import AwsToDf
from newtools import PandasDoggo
import time
from sklearn.model_selection import StratifiedShuffleSplit
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
    df= df.drop(['air_date_time'], axis=1)
    print(df.head())
## Creating season
    series['season']=series.groupby('name')['series_no'].rank()
    series=series.rename(columns={'name':'series_name', 'series_no':'series_id'})

    #checking that shows have exclusive series_id
    c=series.groupby('series_name')['series_id'].apply(lambda x: x.duplicated().any().sum()).sum()
    series=series.drop(['series_name'], axis=1)
    print(c)
    #c=0, therefore merge on only series_id is fine
    df=pd.merge(df,series,on='series_id',how='left')
    del c
    df.head()
##Summation per household season per season per station per day per time-slot, etc

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

    #deleting non useful variables
    del duplicate_counts

    print(len(new_df))
#creating mart_id, market_name, hh_id
    mrkt_hh=new_df[['hh_id','market_id','market_name']].copy()
#dropping hh char
    new_df=new_df.drop(['market_id','market_name','view_duration','programme_duration','genre_name'], axis=1)
    print(len(new_df))
    new_df.head()
# Reducing dem_nam
    sub = [
    "$20,000 - $29,999",
    "$30,000 - $39,999",
    "$40,000 - $49,999",
    "$50,000 - $74,999",
    "$75,000 - $99,999",
    "$100,000 - $124,999",
    "$125,000 - $149,999",
    "$150,000 - $174,999",
    "$175,000 - $199,999",
    "$200,000 - $249,999",
    "$250,000+",
    "One Child in HH",
    "Three Children in HH",
    "Two Adults in HH",
    "Three Adults in HH",
    "Single-Person HH, Male",
    "Single-Person HH, Female",
    "One Adult with Children in HH",
    "A18-24",
    "A25-34",
    "A35-44",
    "A45-54",
    "A55-64",
    "A65+",
    "A18-44",
    "A18-34",
    "Children Present in HH",
    "African American",
    "Four or More Adults in HH",
    "$0 - $19,999",
    "Hispanic",
    "Asian American",
    "White",
    "A35-64",
    "A35-54",
    "A18-49",
    "A50+",
    "M18-24",
    "M18-34",
    "M18-44",
    "M18-49",
    "M25-49",
    "M35-64",
    "M50+",
    "M65+",
    "W18-24",
    "W18-34",
    "W18-44",
    "W18-49",
    "W25-49",
    "W25-54",
    "W35-64",
    "W50+",
    "W65+",
    "M21-34",
    "M21-24",
    "W21-24",
    "W21-34",
    "W21+",
    "W25-34",
    "W35+",
    "W35-54",
    "W45-54",
    "W55-64",
    "M21+",
    "M25-34",
    "M35+",
    "M35-54",
    "M45-54",
    "M55-64",
    "A21-24",
    "A21-34",
    "A25-49",
    "A35+",
    "W35-44",
    "M35-44",
    "$50,000+",
    "M18+",
    "W18+",
    "A55+",
    "M55+",
    "W55+",
    "Single-Parent HH, Male",
    "Single-Parent HH, Female",
    "Two Adults in HH, Both Female",
    "Two Adults in HH, Both Male",
    "A35-54 and A18-24",
    "A18-54",
    "M18-54",
    "W18-54",
    "A21-49",
    "A25-39",
    "M21-49",
    "M21-54",
    "M25-39",
    "W21-49",
    "W21-54",
    "W25-39"
    ]
    print(len(sub))
    dem_nam = dem_nam[dem_nam['name'].isin(sub)]
    dem_nam.head()
    print(len(dem_nam))
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
    print(len(hh_lst))
## Extracting dem_characteristics for sample
    hh_df = pd.DataFrame(hh_lst, columns=['hh_id'])
    dem_ref= hh_df.merge(dem_df, on='hh_id', how='inner')
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
    del sub
# Reducing to 3000 households based on stratified sampling
    transformed_columns = segment_df.columns[1:]
    segment_df['combination'] = segment_df[transformed_columns].apply(lambda x: tuple(x), axis=1)
    uni_seg_combs = segment_df['combination'].unique()
    print(len(uni_seg_combs))

    # Selecting the first instance of the unique dem_id combs
    f = segment_df.columns[0]
    l = segment_df.columns[-1]
    c1 = segment_df[[f, l]]
    cc = c1[c1['hh_id'].isin(sel_ids)].groupby('combination').first().reset_index()

    # Shuffling and selecting 3000

    sel_ids = cc['hh_id'].sample(frac=1).iloc[:3000]
    new_df = new_df[new_df['hh_id'].isin(sel_ids)]
    print(len(new_df))
    segment_df = segment_df[segment_df['hh_id'].isin(sel_ids)]
    del f,l,c1,cc

## Household and market_name
    #Filtering hh_id in mrkt_hh
    hh_lst = segment_df['hh_id'].unique()
    hh_df = pd.DataFrame(hh_lst, columns=['hh_id'])
    mrkt_hh = hh_df.merge(mrkt_hh, on= 'hh_id', how= 'inner').drop_duplicates()

    #Checking if any hh_id has more than 1 market_id
    multi_mrkt = (mrkt_hh.groupby('hh_id')['market_id'].size() > 1).any()
    #As False, dummy tranformation is left to later

## Saving segment_df and mrkt_hh

    doggo = PandasDoggo()
    path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/segment.csv"
    doggo.save(segment_df, path)

    path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/market_hh.csv"
    doggo.save(mrkt_hh, path)
    print("Saved 2")
#Filtering new_df acc hh_id in segment_df
    new_df = new_df[new_df['hh_id'].isin(hh_lst)]
    print(len(new_df))

#Adding 0/1 obs
    del multi_mrkt
    hh_lst = sorted(hh_lst)
    start_time = time.time()

    # Selecting first and last unique show observations and adding identifier column
    c_df = new_df.drop(['hh_id', 'watched'], axis=1)
    c1 = c_df.groupby('series_name').first().reset_index()

    print(len(c1))
    #Stratified sampling of 100 unique shows (from c1) according to genre_id


    # Stratified sampling requires >1 obs in each category -----limitation
    c1 = c1.drop(c1[c1['genre_id'] == 14].index)
    c1 = c1.drop(c1[(c1['genre_id'] == 34) | (c1['genre_id'] == 23)].index)

    sampled_shows = pd.DataFrame()
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=536, random_state=42)
    for train_index, _ in stratified_split.split(c1, c1['genre_id']):
        sampled_shows = c1.iloc[train_index]


    comb_df1 = sampled_shows[:63]
    comb_df2 = sampled_shows[64:]

    print(comb_df2.head())
    # Adding identifier
    comb_df1['temp_id'] = range(len(comb_df1))
    comb_df2['temp_id'] = range(len(comb_df2))

    merged_df1 = pd.merge(new_df, comb_df1, on=['day', 'time', 'network_id', 'station_id',
                                                  'series_name', 'season', 'is_latest_season',
                                                  'genre_id',  'is_national'], how='left')

    merged_df2 = pd.merge(new_df, comb_df2, on=['day', 'time', 'network_id', 'station_id',
                                                  'series_name', 'season', 'is_latest_season',
                                                  'genre_id',  'is_national'], how='left')

    merged_df2.head()
    del dem_dum, dem_nam, fin_dem, df2, series

    for hh_id in hh_lst:
        iteration_start_time = time.time()

        if hh_id <= hh_lst[int(len(hh_lst)/2)]:
        # Filter merged_df by household ID
            temp_df = merged_df1[merged_df1['hh_id'] == hh_id]

        # Getting the distinct list of temp_ids for that hh_id in temp_df
            distinct_temp_ids = temp_df['temp_id'].unique()

        # Getting missing comb from comb_df - combinations of temp
            missing_comb = comb_df1[~comb_df1['temp_id'].isin(distinct_temp_ids)]

        # Adding columns 'hh_id' and 'watched' to missing_comb
            missing_comb['hh_id'] = hh_id
            missing_comb['watched'] = 0

        # Ordering columns according to new_df
            missing_comb = missing_comb[new_df.columns]

        # Concatenating
            new_df = pd.concat([new_df, missing_comb], ignore_index=True)

        elif hh_id > hh_lst[int(len(hh_lst)/2)]:
            # Filter merged_df by household ID
            temp_df = merged_df2[merged_df2['hh_id'] == hh_id]

            # Getting the distinct list of temp_ids for that hh_id in temp_df
            distinct_temp_ids = temp_df['temp_id'].unique()

            # Getting missing comb from comb_df - combinations of temp
            missing_comb = comb_df2[~comb_df2['temp_id'].isin(distinct_temp_ids)]

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
    print(f"Total Number of Combinations: {len(comb_df1)}")
    print('N')
## saving new_df on s3
    path="s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/ML_df3.csv"
    doggo.save(new_df, path)

##checking if different networks have same show
    df3=df.groupby('series_name')['network_id'].nunique()
    df3.head()
    # Merge
    n_df = new_df.merge(segment_df, on = "hh_id", how = 'left')
    n_df = n_df.merge(mrkt_hh, on = 'hh_id', how = 'left')
    n_df = n_df.drop(['market_name','series_name'], axis = 1)
















if __name__ == '__main__':
    run_program('PyCharm')

