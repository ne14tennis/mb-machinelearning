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
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

#MLP
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

#Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

# Re-sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from sklearn.utils import shuffle

def run_program(main):
# Loading reduced dataframes
    atd = AwsToDf()
    X_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_train.csv', 'csv', has_header=True)
    X_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_val.csv', 'csv', has_header=True)
    X_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_test.csv', 'csv', has_header=True)

    y_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_train.csv', 'csv', has_header=True)
    y_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_test.csv', 'csv', has_header=True)
    y_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_val.csv', 'csv', has_header=True)

    print(X_train.head())
    #Dropping Un-named columns
    X_train = X_train.drop(['Unnamed: 0'], axis = 1)
    X_val = X_val.drop(['Unnamed: 0'], axis = 1)
    X_test = X_test.drop(['Unnamed: 0'], axis = 1)

    y_train = y_train.drop(['Unnamed: 0'], axis = 1)
    y_val = y_val.drop(['Unnamed: 0'], axis = 1)
    y_test = y_test.drop(['Unnamed: 0'], axis = 1)

    print("Sets refined")
# Random Under-sampling

    #Seperating majority and minority classes after concatination
    X_y = pd.concat([X_train, y_train], axis=1)
    majority_class_samples = X_y[X_y['watched'] == 0]
    minority_class_samples = X_y[X_y['watched']== 1]
    print(len(minority_class_samples))

    #shuffling and then selecting instances of majority class (same as the len of minority class)
    n = len(minority_class_samples)
    majority_class_samples = shuffle(majority_class_samples, random_state=42)
    majority_class_samples = majority_class_samples[:n]
    len(majority_class_samples)

    #Combining the classes
    n_df = pd.concat([majority_class_samples, minority_class_samples], axis = 0)
    len(n_df)
    #Splitting into X and y
    X_train_rus = n_df.drop(['watched'], axis = 1)
    y_train_rus = n_df['watched']
    print(" RUS conducted")
# Logistic Regression
    # Fitting
    log = LogisticRegression(random_state=77, max_iter=1000)
    log.fit(X_train_rus, y_train_rus)
    #Predicting
    log_y_pred = log.predict(X_test)
    #Prediction Performance Metrics
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
    print("RUS - lOG REG")

# Decision Tree
    # Model Fitting
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_rus, y_train_rus)

    #Predicting
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
    print("RUS - Dt")
#Random forest

    #Fitting
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_rus, y_train_rus)

    #Predicting
    y_pred = rf.predict(X_test)

    #Prediction Metrics
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

# Gradient Boosting Classifier (GBC)
    #Fitting
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train_rus, y_train_rus)
    #Predicting
    y_pred = gbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #Prediction Performance Metrics
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("RUS - BGC")

# XG Boost
    #Fitting
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train_rus, y_train_rus)
    #Predicting
    y_pred = xgb_classifier.predict(X_test)
    #Prediction Performance Metrics
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
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.2f}".format(f1))
    print("F1-score: {:.2f}".format(f1))
# Resampling---ROS

    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    len(X_train_ros)
    print(y_train_ros.value_counts())
    print(X_test.head())
#Decesion Tree

    # Model Fitting
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_ros, y_train_ros)

    #Predicting
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

#Random forest

    #Fitting
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_ros, y_train_ros)

    #Predicting
    y_pred = rf.predict(X_test)

    #Prediction Metrics
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

# Gradient Boosting Classifier (GBC)
    #Fitting
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train_ros, y_train_ros)
    #Predicting
    y_pred = gbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #Prediction Performance Metrics
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("ROS - BGC")

# XG Boost
    #Fitting
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train_ros, y_train_ros)
    #Predicting
    y_pred = xgb_classifier.predict(X_test)
    #Prediction Performance Metrics
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
# Decision Tree
    # Model Fitting
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_sm, y_train_sm)

    #Predicting
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
#Random forest

    #Fitting
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_sm, y_train_sm)

    #Predicting
    y_pred = rf.predict(X_test)

    #Prediction Metrics
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

# Gradient Boosting Classifier (GBC)
    #Fitting
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train_sm, y_train_sm)
    #Predicting
    y_pred = gbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #Prediction Performance Metrics
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("RUS - BGC")
# XG Boost
    #Fitting
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train_sm, y_train_sm)
    #Predicting
    y_pred = xgb_classifier.predict(X_test)
    #Prediction Performance Metrics
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


if __name__ == '__main__':
    run_program('PyCharm')