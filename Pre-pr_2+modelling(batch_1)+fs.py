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

# Libraries for Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#MLP
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

#Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

class a2:
    def p2(self):
        # Loading refined dataframes
        atd = AwsToDf()
        new_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'ML_df3.csv', 'csv', has_header=True)
        segment_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'segment.csv', 'csv', has_header=True)
        mrkt_hh = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'market_hh.csv', 'csv', has_header=True)

        print('check 1')
        # Converting time to hour of day
        new_df['time'] = pd.to_datetime(new_df['time'])
        new_df['hour_of_day'] = new_df['time'].dt.hour
        new_df = new_df.drop(['time', 'Unnamed: 0'], axis=1)
        print(new_df.head())
        # Merge
        n_df = new_df.merge(segment_df, on="hh_id", how='left')
        n_df = n_df.merge(mrkt_hh, on='hh_id', how='left')
        n_df = n_df.drop(['market_name', 'series_name'], axis=1)
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

        # Dropping and concating
        n_df = n_df.drop(['hh_id', 'day', 'network_id', 'station_id', 'season', 'genre_id', 'market_id', 'combination'],
                         axis=1)
        n_df = pd.concat(
            [n_df, day_dummy, hh_dummy, network_dummy, station_dummy, season_dummy, genre_dummy, mrkt_dummy], axis=1)

        n_df.head()
        print(len(n_df))
        # Removing unnamed columns
        n_df = n_df.drop(['Unnamed: 0_y', 'Unnamed: 0_x'], axis=1)

        n_df.head()

        # Converting bool into 0/1 for MLP processing
        n_df['is_latest_season'] = n_df['is_latest_season'].astype(int)
        n_df['is_national'] = n_df['is_national'].astype(int)

        print(n_df.head)
        # Splitting into train and test---For Models requiring scaling
        # Split data into X (features) and y (target variable)
        X = n_df.drop(['watched', 'hour_of_day'], axis=1)
        y = n_df['watched']

        X.columns = X.columns.astype(str)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77, stratify=y)
        X_train.head()
        # Modelling

        # Logistic Regression

        log = LogisticRegression(random_state=77, max_iter=1000)
        log.fit(X_train, y_train)
        #Predicting
        log_y_pred = log.predict(X_test)
        #Prediction performance metrics
        accuracy = accuracy_score(y_test, log_y_pred)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, log_y_pred)
        precision = precision_score(y_test, log_y_pred)
        recall = recall_score(y_test, log_y_pred)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, log_y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # For Models not requiring scaling
        del new_df
        # Splitting data into X (features) and y (target variable)
        X = n_df.drop(['watched', 'hour_of_day_scaled'], axis=1)
        y = n_df['watched']
        X.columns = X.columns.astype(str)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77, stratify=y)
        X_train.head()
        # Deleting un-necessary variables
        del day_dummy, genre_dummy, hh_dummy, mrkt_dummy, mrkt_hh, network_dummy, season_dummy, segment_df, station_dummy
        # Splitting data into train and val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                          stratify=y_train)

        print("Train/val split complete")
        # KNN Classifier --- eliminated due to requirement of excess memory

        # Decision Tree Classifier

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
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

        # RF Classifier

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
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

        # XG Boost

        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Precision:", precision)
        print("F1 score:", f1)
        print("Recall:", recall)

        print("Shape of X:", X.shape)
        print("Shape of y:", y.shape)

        print("size check")

        # MLP
        # Set random seed
        tf.random.set_seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation fn
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)

        # Evaluate the model on the test set
        loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
        print("Binary Cross-Entropy Loss:", loss)
        print("Binary Accuracy on test set:", binary_accuracy)
        print("Precision on test set:", precision)
        print("Recall on test set:", recall)

        # Make predictions on the test set
        binary_preds = model.predict(X_test)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_test, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, binary_preds_rounded))
        print("F1-score: {:.2f}".format(f1))
        print("F1-score: {:.2f}".format(f1))
        # Detection of model fit given the set hyperparameters

        # Plot the loss curve for training and validation
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        print("Overfit Detected")

        # Univariate Feature Selection
        # Results checked with k=150,200,300,500,800. Best when k=300, hence chosen

        k = 300
        selector = SelectKBest(chi2, k=k)

        # to know which 300 were selected
        selected_feature_ind = selector.get_support(indices=True)
        selected_feature_nam = X_train.columns[selected_feature_ind]

        # reducing to 300 features, across X_train,X_test, X_val
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)

        print("Univariate FS")
        # MLP
        tf.random.set_seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation fn
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        # Train the model
        history = model.fit(X_train_selected, y_train, epochs=50, batch_size=36,
                            validation_data=(X_val_selected, y_val),
                            verbose=2)

        # Evaluate the model on the test set
        loss, binary_accuracy, precision, recall = model.evaluate(X_test_selected, y_test)
        print("Binary Cross-Entropy Loss:", loss)
        print("Binary Accuracy on test set:", binary_accuracy)
        print("Precision on test set:", precision)
        print("Recall on test set:", recall)

        # Make predictions on the test set
        binary_preds = model.predict(X_test_selected)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_test, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, binary_preds_rounded))
        print("F1-score: {:.2f}".format(f1))
        print(len(binary_preds_rounded))
        # Saving X_train_selected, X_test_selected, X_val_selected, y_train, y_val, y_test
        doggo = PandasDoggo()

        X_train_selected = pd.DataFrame(X_train_selected)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/X_train.csv"
        doggo.save(X_train_selected, path)
        X_test_selected = pd.DataFrame(X_test_selected)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/X_test.csv"
        doggo.save(X_test_selected, path)
        X_val_selected = pd.DataFrame(X_val_selected)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/X_val.csv"
        doggo.save(X_val_selected, path)

        y_train = pd.DataFrame(y_train)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/y_train.csv"
        doggo.save(y_train, path)
        y_test = pd.DataFrame(y_test)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/y_test.csv"
        doggo.save(y_test, path)
        y_val = pd.DataFrame(y_val)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/y_val.csv"
        doggo.save(y_val, path)

        print("Saved X and y splits complete")

        print("Part B Complete")