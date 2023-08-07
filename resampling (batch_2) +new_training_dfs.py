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

# Libraries for Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# MLP
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

# Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

# Re-sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

class a2:
    def resampling(self):
        # Loading reduced dataframes
        atd = AwsToDf()
        X_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_train.csv', 'csv', has_header=True)
        X_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_val.csv', 'csv', has_header=True)
        X_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_test.csv', 'csv', has_header=True)

        y_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_train.csv', 'csv', has_header=True)
        y_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_test.csv', 'csv', has_header=True)
        y_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_val.csv', 'csv', has_header=True)

        print(X_train.head())
        # Dropping Un-named columns
        X_train = X_train.drop(['Unnamed: 0'], axis=1)
        X_val = X_val.drop(['Unnamed: 0'], axis=1)
        X_test = X_test.drop(['Unnamed: 0'], axis=1)

        y_train = y_train.drop(['Unnamed: 0'], axis=1)
        y_val = y_val.drop(['Unnamed: 0'], axis=1)
        y_test = y_test.drop(['Unnamed: 0'], axis=1)

        print("Sets refined")

        # Resampling---ROS

        ros = RandomOverSampler(random_state=42)
        X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        len(X_train_ros)
        print(y_train_ros.value_counts())
        print(X_test.head())
        # Logistic Regression

        # Fitting
        log = LogisticRegression(random_state=77, max_iter=1000)
        log.fit(X_train_ros, y_train_ros)

        # Predicting (on val)
        log_y_pred = log.predict(X_val)
        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, log_y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, log_y_pred), 5)
        precision = round(precision_score(y_val, log_y_pred), 5)
        recall = round(recall_score(y_val, log_y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, log_y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROS - lOG REG")

        # Decision Tree

        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_ros, y_train_ros)

        # Predicting on val
        y_pred = dt.predict(X_val)

        # Performance metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROS - Dt")

        # Random Forest
        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_ros, y_train_ros)

        # Predicting (on val)
        y_pred = rf.predict(X_val)

        # Prediction Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROS - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_ros, y_train_ros)

        # Predicting (on val)
        y_pred = xgb_classifier.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("F1 score:", f1)
        print("Recall:", recall)

        print("ROS - XG Boost")

        # MLP
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_ros.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and metrics
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()])

        # Train the model
        history = model.fit(X_train_ros, y_train_ros, epochs=50, batch_size=36, validation_data=(X_val, y_val),
                            verbose=2)

        # Evaluate the model on the validation set
        loss, binary_accuracy, precision, recall, accuracy = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 5))
        print("Binary Accuracy on validation set:", round(binary_accuracy, 5))
        print("Precision on validation set:", round(precision, 5))
        print("Recall on validation set:", round(recall, 5))
        print("Accuracy on validation set:", round(accuracy, 5))

        # Make predictions on the validation set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("ROS- MLP")

        # SMOTHE
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        len(X_train_sm)
        print(y_train_sm.value_counts())

        print("SMOTHE")
        # Logistic Regression
        # Fitting
        log = LogisticRegression(random_state=77, max_iter=1000)
        log.fit(X_train_sm, y_train_sm)

        # Predicting (on val)
        log_y_pred = log.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, log_y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, log_y_pred), 5)
        precision = round(precision_score(y_val, log_y_pred), 5)
        recall = round(recall_score(y_val, log_y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, log_y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("SMOTHE - lOG REG")

        # Decision Tree

        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_sm, y_train_sm)

        # Predicting on val
        y_pred = dt.predict(X_val)

        # Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("SMOTHE - Dt")

        # Random Forest
        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_sm, y_train_sm)

        # Predicting (on val)
        y_pred = rf.predict(X_val)

        # Prediction Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("SMOTHE - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_sm, y_train_sm)

        # Predicting (on val)
        y_pred = xgb_classifier.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("F1 score:", f1)
        print("Recall:", recall)

        print("SMOTHE - XG Boost")
        # MLP
        # Set random seed
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_sm.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and metrics
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()])

        # Train the model
        history = model.fit(X_train_sm, y_train_sm, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)

        # Evaluate the model on the validation set
        loss, binary_accuracy, precision, recall, accuracy = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 5))
        print("Binary Accuracy on validation set:", round(binary_accuracy, 5))
        print("Precision on validation set:", round(precision, 5))
        print("Recall on validation set:", round(recall, 5))
        print("Accuracy on validation set:", round(accuracy, 5))

        # Make predictions on the validation set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("SMOTHE- MLP")

        # Random Under-sampling

        # Seperating majority and minority classes after concatination
        X_y = pd.concat([X_train, y_train], axis=1)
        majority_class_samples = X_y[X_y['watched'] == 0]
        minority_class_samples = X_y[X_y['watched'] == 1]
        print(len(minority_class_samples))

        # shuffling and then selecting instances of majority class (same as the len of minority class)
        n = len(minority_class_samples)
        majority_class_samples = shuffle(majority_class_samples, random_state=42)
        majority_class_samples = majority_class_samples[:n]
        len(majority_class_samples)

        # Combining the classes
        n_df = pd.concat([majority_class_samples, minority_class_samples], axis=0)
        len(n_df)
        # Splitting into X and y
        X_train_rus = n_df.drop(['watched'], axis=1)
        y_train_rus = n_df['watched']
        print("RUS conducted")
        # Fitting
        log_rus = LogisticRegression(random_state=77, max_iter=1000)
        log_rus.fit(X_train_rus, y_train_rus)

        # Predicting (on val)
        y_pred_log = log_rus.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred_log), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred_log), 5)
        precision = round(precision_score(y_val, y_pred_log), 5)
        recall = round(recall_score(y_val, y_pred_log), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred_log))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("RUS - lOG REG")

        # Decision Tree

        # Model Fitting
        dt_rus = DecisionTreeClassifier(random_state=42)
        dt_rus.fit(X_train_rus, y_train_rus)

        # Predicting on val
        y_pred_dt = dt_rus.predict(X_val)

        # Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred_dt), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred_dt), 5)
        precision = round(precision_score(y_val, y_pred_dt), 5)
        recall = round(recall_score(y_val, y_pred_dt), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred_dt))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("RUS - Dt")

        # Random Forest
        # Fitting
        rf_rus = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_rus.fit(X_train_rus, y_train_rus)

        # Predicting (on val)
        y_pred = rf_rus.predict(X_val)

        # Prediction Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("RUS - rf")

        # XG Boost
        # Fitting
        xgb_classifier_rus = xgb.XGBClassifier()
        xgb_classifier_rus.fit(X_train_rus, y_train_rus)

        # Predicting (on val)
        y_pred = xgb_classifier_rus.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("F1 score:", f1)
        print("Recall:", recall)

        print("RUS - XG Boost")
        # MLP
        # Set random seed
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)

        # Model architecture
        model_rus = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_rus.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and metrics
        model_rus.compile(loss='binary_crossentropy', optimizer=optimizer,
                          metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()])

        # Train the model
        history = model_rus.fit(X_train_rus, y_train_rus, epochs=50, batch_size=36, validation_data=(X_val, y_val),
                                verbose=2)

        # Evaluate the model on the validation set
        loss, binary_accuracy, precision, recall, accuracy = model_rus.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 5))
        print("Binary Accuracy on validation set:", round(binary_accuracy, 5))
        print("Precision on validation set:", round(precision, 5))
        print("Recall on validation set:", round(recall, 5))
        print("Accuracy on validation set:", round(accuracy, 5))

        # Make predictions on the validation set
        binary_preds = model_rus.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("RUS- MLP")
        # Combining ROS and RUS

        # oversampling minority to increase it from 48.16% to 65% of majority class
        over = RandomOverSampler(sampling_strategy=0.65)
        # fit and apply the transform
        X_train_a, y_train_a = over.fit_resample(X_train, y_train)
        y_train_a.value_counts()
        print("Oversampling to 65% of majority class conducted")

        # Undersampling majority class to bring it to 65% of its original total

        # Seperating majority and minority classes after concatination
        X_y_c = pd.concat([X_train_a, y_train_a], axis=1)
        majority_class_s = X_y_c[X_y_c['watched'] == 0]
        minority_class_s = X_y_c[X_y_c['watched'] == 1]
        print(len(minority_class_s))

        # shuffling and then selecting instances of majority class (same as the len of minority class)
        c_n = len(minority_class_s)
        majority_class_s = shuffle(majority_class_s, random_state=42)
        majority_class_s = majority_class_s[:c_n]
        len(majority_class_s)

        # Combining the classes
        c_df = pd.concat([majority_class_s, minority_class_s], axis=0)
        len(c_df)
        # Splitting into X and y
        X_train_comb = c_df.drop(['watched'], axis=1)
        y_train_comb = c_df['watched']

        print("Comb training sets created")

        # Fitting
        log = LogisticRegression(random_state=77, max_iter=1000)
        log.fit(X_train_comb, y_train_comb)

        # Predicting (on val)
        log_y_pred = log.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, log_y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, log_y_pred), 5)
        precision = round(precision_score(y_val, log_y_pred), 5)
        recall = round(recall_score(y_val, log_y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, log_y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Comb - lOG REG")

        # Decision Tree

        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_comb, y_train_comb)

        # Predicting on val
        y_pred = dt.predict(X_val)

        # Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Comb - Dt")

        # Random Forest
        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_comb, y_train_comb)

        # Predicting (on val)
        y_pred = rf.predict(X_val)

        # Prediction Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Comb - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_comb, y_train_comb)

        # Predicting (on val)
        y_pred = xgb_classifier.predict(X_val)

        # Prediction Performance Metrics
        accuracy = round(accuracy_score(y_val, y_pred), 5)
        print("Accuracy:", accuracy)
        f1 = round(f1_score(y_val, y_pred), 5)
        precision = round(precision_score(y_val, y_pred), 5)
        recall = round(recall_score(y_val, y_pred), 5)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Precision:", precision)
        print("F1 score:", f1)
        print("Recall:", recall)

        print("Comb - XG Boost")
        # MLP
        # Set random seed
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_comb.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and metrics
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()])

        # Train the model
        history = model.fit(X_train_comb, y_train_comb, epochs=50, batch_size=36, validation_data=(X_val, y_val),
                            verbose=2)

        # Evaluate the model on the validation set
        loss, binary_accuracy, precision, recall, accuracy = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 5))
        print("Binary Accuracy on validation set:", round(binary_accuracy, 5))
        print("Precision on validation set:", round(precision, 5))
        print("Recall on validation set:", round(recall, 5))
        print("Accuracy on validation set:", round(accuracy, 5))

        # Make predictions on the validation set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("Comb- MLP")

        # Best model for voting classifier------RUS

        # Performance of slected resampling method (rus through validation comparison)
        y_pred_log_rus = log_rus.predict(X_test)
        y_pred_dt_rus = dt_rus.predict(X_test)
        y_pred_rf_rus = rf_rus.predict(X_test)
        y_pred_xg_rus = xgb_classifier_rus.predict(X_test)
        y_pred_MLP_rus = model_rus.predict(X_test)
        y_pred_MLP_rus = [1 if pred > 0.5 else 0 for pred in y_pred_MLP_rus]
        # Selected training data- RUS

        # Comparing Models through confusion matrix heatmap
        plt.figure(figsize=(16, 7))

        # Log Regression model
        plt.subplot(2, 3, 1)
        sns.heatmap(confusion_matrix(y_test, y_pred_log_rus) / np.sum(confusion_matrix(y_test, y_pred_log_rus)),
                    annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('Logistic')

        # Decision Tree confusion matrix heatmap
        plt.subplot(2, 3, 2)
        sns.heatmap(confusion_matrix(y_test, y_pred_dt_rus) / np.sum(confusion_matrix(y_test, y_pred_dt_rus)),
                    annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('Decision Tree')

        # Random Forest
        plt.subplot(2, 3, 3)
        sns.heatmap(confusion_matrix(y_test, y_pred_rf_rus) / np.sum(confusion_matrix(y_test, y_pred_rf_rus)),
                    annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('Random Forest')

        # XGBoost
        plt.subplot(2, 3, 4)
        sns.heatmap(confusion_matrix(y_test, y_pred_xg_rus) / np.sum(confusion_matrix(y_test, y_pred_xg_rus)),
                    annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('XGBoost')

        # MLP
        plt.subplot(2, 3, 5)
        sns.heatmap(confusion_matrix(y_test, y_pred_MLP_rus) / np.sum(confusion_matrix(y_test, y_pred_MLP_rus)),
                    annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('MLP')

        plt.show();
        print("Confusion Matrix")

        print("Confusion Matrix")
    # Saving the rus training data as X_train_nw , y_train_nw

        doggo = PandasDoggo()
        X_train_nw = pd.DataFrame(X_train_rus)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/X_train_nw.csv"
        doggo.save(X_train_nw, path)

        y_train_nw = pd.DataFrame(y_train_rus)
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/y_train_nw.csv"
        doggo.save(y_train_nw, path)

        print("Save complete")

