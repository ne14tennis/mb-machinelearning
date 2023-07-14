# Basic
import numpy as np
import pandas as pd
# Loading, converting, saving libraries
from aws_to_df import AwsToDf
from newtools import PandasDoggo

# Libraries for Plots/Data Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.inspection import permutation_importance

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
    new_df = new_df.drop(['time', 'Unnamed: 0'], axis=1)
    print(new_df.head())
    # Merge
    n_df = new_df.merge(segment_df, on = "hh_id", how = 'left')
    n_df = n_df.merge(mrkt_hh, on = 'hh_id', how = 'left')
    n_df = n_df.drop(['market_name','series_name'], axis = 1)
    print(len(n_df))
    n_df.isna().sum()
# Dummy Dataframe
    # Getting dummies -----OHE
    day_dummy = pd.get_dummies(n_df.day)
    hh_dummy = pd.get_dummies(n_df.hh_id)
    network_dummy = pd.get_dummies(n_df.network_id)
    station_dummy = pd.get_dummies(n_df.station_id)
    season_dummy = pd.get_dummies(n_df.season)
    genre_dummy = pd.get_dummies(n_df.genre_id)
    mrkt_dummy = pd.get_dummies(n_df.market_id)

    print('Dummies created')

    #Dropping and concating
    n_df = n_df.drop(['hh_id','day','network_id','station_id','season','genre_id','market_id','combination'], axis = 1)
    n_df = pd.concat([n_df, day_dummy, hh_dummy, network_dummy, station_dummy, season_dummy, genre_dummy, mrkt_dummy], axis = 1)

    n_df.head()
    print(len(n_df))
    # Removing unnamed columns
    n_df = n_df.drop(['Unnamed: 0_y','Unnamed: 0_x'], axis = 1)

    n_df.head()


    print("Saved t_df")
# Splitting into train and test
    # Split data into X (features) and y (target variable)
    X = n_df.drop(['watched'], axis=1)
    y = n_df['watched']

    X.columns = X.columns.astype(str)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)
    X_train.head()
# Modelling

    # Logistic Regression

    # SVM

    #SGD Classifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    # Decision Tree Classifier

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    # RF Classifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Random Forest (SMOTHE)


   # Gradient Boosting Classifier

    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)


    # XG Boost

if __name__ == '__main__':
    run_program('PyCharm')

