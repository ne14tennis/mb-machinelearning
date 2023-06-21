import numpy as np
import pandas as pd
from aws_to_df import AwsToDf

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
## coverting to datetime data type for date
    df['date']= pd.to_datetime(df['date'])

##extracting day from datetime
    df['day']=df['date'].dt.day_name()
    df=df.drop(['date'], axis=1)

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
#rounding-up time

##Summation per season per series
    ##checking same series, different epsiode_id
    c1=df.groupby(['series_name','series_id','episode_id']).size().reset_index(name='count')
    ## multiple found, therefore summation on all other columns
    ### dropping episode_id
    new_df=df.drop(['episode_id'],axis=1)
    new_df = new_df.groupby(['hh_id', 'day','time','market_name', 'network_id', 'station_id', 'series_name', 'series_id', 'is_latest_season', 'genre_name',
     'genre_id', 'is_national', 'market_id']).agg({'view_duration': 'sum', 'programme_duration': 'sum'}).reset_index()
    print(new_df.head())

## checking for duplicate combinations in each household

    ## checking observations where view_duration>programee_duration
    df2=df[df['programme_duration']<df['view_duration']]
    ## as a percentage
    print("proportion="+ str(len(df2)/len(df)))

    ##checking if different networks have same show
    df3=df.groupby('series_name')['network_id'].nunique()
    df3.head()
    ## there are shows which are screened by multiple networks (list-netwotk_id+series)
    ## are they different shows or the same show screened by multiple networks?
    df3 = df.groupby('series_name')['network_id'].unique()
    df3.head()
    ### checking for combinations across series, network, station
    comb = df.groupby(['series_name', 'network_id', 'station_id']).size().reset_index(name='count')
    comb.head()
    ## different comb exists----(list-series+network+station)





if __name__ == '__main__':
    run_program('PyCharm')

