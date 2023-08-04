# Basic
import pandas as pd
import numpy as np

# Loading, converting, saving libraries
from aws_to_df import AwsToDf
from newtools import PandasDoggo

# Libraries for Plots/Data Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
# MLP
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

# Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

# Tuning
from sklearn.model_selection import GridSearchCV

def run_program(main):

    # Loading dfs
    atd = AwsToDf()
    X_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_train_nw.csv', 'csv', has_header=True)
    X_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_val.csv', 'csv', has_header=True)
    X_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_test.csv', 'csv', has_header=True)

    y_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_train_nw.csv', 'csv', has_header=True)
    y_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_test.csv', 'csv', has_header=True)
    y_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_val.csv', 'csv', has_header=True)

    print("Re-sampled data loaded")
# Dropping Un-named
    X_train = X_train.drop(['Unnamed: 0'], axis=1)
    y_train = y_train.drop(['Unnamed: 0'], axis=1)
    X_test = X_test.drop(['Unnamed: 0'], axis=1)
    y_test = y_test.drop(['Unnamed: 0'], axis=1)
    X_val = X_val.drop(['Unnamed: 0'], axis=1)
    y_val = y_val.drop(["Unnamed: 0"], axis = 1)

    print(X_train.columns)
# Fine-tuning

    # Random Forest

    params = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [25, 50, 75, 100,  None],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2, 3],
    }

    rfc = RandomForestClassifier(random_state = 42)
    rf_clf = GridSearchCV(estimator = rfc,
                          param_grid = params,
                          scoring = 'accuracy',
                          verbose = 1,
                          n_jobs = -1)

    rf_clf.fit(X_train, y_train)

    rf_clf.best_estimator_

    rf_clf.best_params_


    rf = RandomForestClassifier(n_estimators=100, max_depth=7, max_features='auto', min_samples_leaf=1,
                                min_samples_split=2, criterion='gini', random_state=42)
    rf.fit(X_train, y_train)

    # Predicting
    y_pred = rf.predict(X_test)

    # Prediction Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Best params for RF found")

    #XG Boost
    """
    params = {'max_depth': [3, 4, 5, 6, 7],
              'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.3],
              'n_estimators': [50, 100, 300, 500, 750],
              'colsample_bytree': [0.1, 0.3, 0.5, 1.0]}
    xgbc = XGBClassifier(seed= 77)
    xg_clf = GridSearchCV(estimator=xgbc,
                       param_grid=params,
                       scoring='accuracy',
                       verbose=1)
    xg_clf.fit(X_train, y_train)

    xg_clf.best_estimator_

    xg_clf.best_params_
    """
    xg = XGBClassifier(random_state=77, colsample_bytree=0.3, learning_rate=0.3, max_depth=6,
                       n_estimators=750)

    xg.fit(X_train, y_train)

    # Prediction on test set
    xg_y_pred = xg.predict(X_test)

    # Prediction Performance Metrics
    accuracy = accuracy_score(y_test, xg_y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, xg_y_pred)
    precision = precision_score(y_test, xg_y_pred)
    recall = recall_score(y_test, xg_y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, xg_y_pred))
    print("Precision:", precision)
    print("F1 score:", f1)
    print("Recall:", recall)
    print(confusion_matrix(y_test, xg_y_pred))
    print(" Best params for XG Boost found")

#MLP


if __name__ == '__main__':
    run_program('PyCharm')