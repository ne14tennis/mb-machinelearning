# Basic
import numpy as np
import pandas as pd
# Loading, converting, saving libraries
from aws_to_df import AwsToDf
from newtools import PandasDoggo
# Scaling
from sklearn.preprocessing import MinMaxScaler
# Libraries for Plots/Data Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries Modelling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

#MLP
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Input, Concatenate

#Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.inspection import permutation_importance

# Learning Curve
from sklearn.model_selection import learning_curve

# Cross-val
from sklearn.model_selection import cross_val_score


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

# Manipulating for dummy conversion and reading
    n_df['hh_id'] = 'h ' + n_df['hh_id'].astype(str)
    n_df['network_id'] = 'n ' + n_df['network_id'].astype(str)
    n_df['station_id'] = 'st ' + n_df['station_id'].astype(str)
    n_df['genre_id'] = 'g ' + n_df['genre_id'].astype(str)
    n_df['market_id'] = 'mk ' + n_df['market_id'].astype(str)

    print(n_df.head())


# Scaling hour of day
    scaler = MinMaxScaler()
    hod_scaled = scaler.fit_transform(n_df['hour_of_day'].values.reshape(-1, 1))
    n_df['hour_of_day_scaled'] = hod_scaled
    print(n_df.head())

# Checking for unique categories


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

    # For Models requiring scaling
    # Split data into X (features) and y (target variable)
    X = n_df.drop(['watched','hour_of_day'], axis=1)
    y = n_df['watched']

    X.columns = X.columns.astype(str)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77, stratify = y )
    X_train.head()
# Modelling

    # Logistic Regression
    """
    log = LogisticRegression(random_state=77, max_iter=1000)

    log.fit(X_train, y_train)

    # Predict the response for test dataset

    log_y_pred = log.predict(X_test)
    print(classification_report(y_test, log_y_pred))
    accuracy = accuracy_score(y_test, log_y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, log_y_pred)
    precision = precision_score(y_test, log_y_pred)
    recall = recall_score(y_test, log_y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    """

    # SVM
    """
    svm = SVC(kernel='rbf', random_state=1)
    svm.fit(X_train, y_train)
    svm_y_pred = svm.predict(X_test)
    print(classification_report(y_test, svm_y_pred))
    accuracy = accuracy_score(y_test, svm_y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, svm_y_pred)
    precision = precision_score(y_test, svm_y_pred)
    recall = recall_score(y_test, svm_y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    """

    # For Models not requiring scaling
    # Split data into X (features) and y (target variable)
    X = n_df.drop(['watched','hour_of_day_scaled'], axis=1)
    y = n_df['watched']
    X.columns = X.columns.astype(str)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77, stratify=y)
    X_train.head()

    """
    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    """

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

    rf = RandomForestClassifier(n_estimators= 100, random_state=42)
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


   # Gradient Boosting Classifier (GBC)
    """

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

    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("F1 score:", f1)
    print("Recall:", recall)

    """

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    print("size check")

    """
    
    k=5
    
    train_scores = cross_val_score(rf, X_train, y_train, cv=k)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, k + 1), train_scores, label='Cross-Validated Training Score', marker='o')
    plt.axhline(np.mean(train_scores), color='red', linestyle='--', label='Average Training Score')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross-Validated Training Score vs. Fold')
    plt.legend()
    plt.grid()
    plt.show()

    # MLP
    
    
   
    """

    print(len(X_train))

    # Learning Curve

    train_sizes, train_scores, val_scores = learning_curve(rf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5),
                                                           scoring='accuracy')

    # Calculate the mean and standard deviation for training and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation Accuracy', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    print(len(X_test))

    print(len(X_train))


if __name__ == '__main__':
    run_program('PyCharm')

