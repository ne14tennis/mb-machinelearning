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
        # Predicting
        log_y_pred = log.predict(X_test)
        # Prediction Performance Metrics
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
        print("ROS - lOG REG")

        # Decesion Tree

        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_ros, y_train_ros)

        # Predicting
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
        print("ROS - Dt")

        # Random forest

        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_ros, y_train_ros)

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
        print("ROS - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_ros, y_train_ros)
        # Predicting
        y_pred = xgb_classifier.predict(X_test)
        # Prediction Performance Metrics
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

        print("ROS - XG Boost")

        # MLP
        # Set random seed
        tf.random.set_seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_ros.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        # Train the model
        history = model.fit(X_train_ros, y_train_ros, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)

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
        # Predicting
        log_y_pred = log.predict(X_test)
        # Prediction Performance Metrics
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
        print("SMOTHE - lOG REG")

        # Decision Tree
        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_sm, y_train_sm)

        # Predicting
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
        print("SMOTHE - Dt")
        # Random forest

        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_sm, y_train_sm)

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
        print("RUS - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_sm, y_train_sm)
        # Predicting
        y_pred = xgb_classifier.predict(X_test)
        # Prediction Performance Metrics
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

        print("RUS - XG Boost")
        # MLP
        tf.random.set_seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_sm.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        # Train the model
        history = model.fit(X_train_sm, y_train_sm, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)

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
        # Logistic Regression
        # Fitting
        log = LogisticRegression(random_state=77, max_iter=1000)
        log.fit(X_train_rus, y_train_rus)
        # Predicting
        y_pred_log = log.predict(X_test)
        # Prediction Performance Metrics
        accuracy = accuracy_score(y_test, y_pred_log)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, y_pred_log)
        precision = precision_score(y_test, y_pred_log)
        recall = recall_score(y_test, y_pred_log)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_log))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("RUS - lOG REG")

        # Decision Tree
        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_rus, y_train_rus)

        # Predicting
        y_pred_dt = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_dt)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, y_pred_dt)
        precision = precision_score(y_test, y_pred_dt)
        recall = recall_score(y_test, y_pred_dt)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_dt))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("RUS - Dt")
        # Random forest

        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_rus, y_train_rus)

        # Predicting
        y_pred_rf = rf.predict(X_test)

        # Prediction Metrics
        accuracy = accuracy_score(y_test, y_pred_rf)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, y_pred_rf)
        precision = precision_score(y_test, y_pred_rf)
        recall = recall_score(y_test, y_pred_rf)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_rf))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("RUS - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_rus, y_train_rus)
        # Predicting
        y_pred_xg = xgb_classifier.predict(X_test)
        # Prediction Performance Metrics
        accuracy = accuracy_score(y_test, y_pred_xg)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, y_pred_xg)
        precision = precision_score(y_test, y_pred_xg)
        recall = recall_score(y_test, y_pred_xg)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_xg))
        print("Precision:", precision)
        print("F1 score:", f1)
        print("Recall:", recall)

        print("RUS - XG Boost")

        # MLP
        tf.random.set_seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_rus.shape[1],)),
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
        history = model.fit(X_train_rus, y_train_rus, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)

        # Evaluate the model on the test set
        loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
        print("Binary Cross-Entropy Loss:", loss)
        print("Binary Accuracy on test set:", binary_accuracy)
        print("Precision on test set:", precision)
        print("Recall on test set:", recall)

        # Make predictions on the test set
        binary_preds = model.predict(X_test)
        # Round the predictions to get the binary class labels (0 or 1)
        y_pred_MLP = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score for binary classification
        f1 = f1_score(y_test, y_pred_MLP)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_MLP))
        print("F1-score: {:.2f}".format(f1))
        print("F1-score: {:.2f}".format(f1))

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
        # Logistic Regression

        # Fitting
        log = LogisticRegression(random_state=77, max_iter=1000)
        log.fit(X_train_comb, y_train_comb)
        # Predicting
        log_y_pred = log.predict(X_test)
        # Prediction Performance Metrics
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
        print("RUS/ROS - lOG REG")

        # Decision Tree
        # Model Fitting
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_comb, y_train_comb)

        # Predicting
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
        print("RUS/ROS - Dt")
        # Random forest

        # Fitting
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_comb, y_train_comb)

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
        print("RUS/ROS - rf")

        # XG Boost
        # Fitting
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train_comb, y_train_comb)
        # Predicting
        y_pred = xgb_classifier.predict(X_test)
        # Prediction Performance Metrics
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

        print("RUS/ROS - XG Boost")

        # MLP
        tf.random.set_seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_comb.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        # Train the model
        history = model.fit(X_train_comb, y_train_comb, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)

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
        # Best model for voting classifier------RUS
        # Best model for only MLP------Comb: RUS/ROS

        # Selected training data- RUS

        # Comparing Models through confusion matrix heatmap
        plt.figure(figsize=(16, 7))

        # Log Regression model
        plt.subplot(2, 3, 1)
        sns.heatmap(confusion_matrix(y_test, y_pred_log) / np.sum(confusion_matrix(y_test, y_pred_log)), annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('Logistic')

        # Decision Tree confusion matrix heatmap
        plt.subplot(2, 3, 2)
        sns.heatmap(confusion_matrix(y_test, y_pred_dt) / np.sum(confusion_matrix(y_test, y_pred_dt)), annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('Decision Tree')

        # Random Forest
        plt.subplot(2, 3, 3)
        sns.heatmap(confusion_matrix(y_test, y_pred_rf) / np.sum(confusion_matrix(y_test, y_pred_rf)), annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('Random Forest')

        # XGBoost
        plt.subplot(2, 3, 4)
        sns.heatmap(confusion_matrix(y_test, y_pred_xg) / np.sum(confusion_matrix(y_test, y_pred_xg)), annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('XGBoost')

        # MLP
        plt.subplot(2, 3, 5)
        sns.heatmap(confusion_matrix(y_test, y_pred_MLP) / np.sum(confusion_matrix(y_test, y_pred_MLP)), annot=True,
                    fmt='.2%', cmap='Blues')
        plt.title('MLP')

        plt.show();
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

