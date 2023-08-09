# Basic
import pandas as pd
import numpy as np

# Loading, converting, saving libraries
from aws_to_df import AwsToDf
from newtools import PandasDoggo

from aws_to_df import AwsToDf


# Libraries for Plots/Data Visualisations
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

# Libraries for Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# MLP
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense

# Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

# Cross Val
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std

# LC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer

# Tuning
from sklearn.model_selection import GridSearchCV

def run_program(main):
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


#MLP model
    # Neurons per layer = 100
    tf.random.set_seed(42)
    np.random.seed(42)
    norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
    # Model architecture
    model = tf.keras.Sequential([
        norm_layer,
        tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Defining the optimizer with a learning rate of 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)

    # Evaluate the model on the train set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the val set
    binary_preds = model.predict(X_val)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    print("MLP Classifier")
# Evaluation of MLP classifier




# Setting precision -recall threshold

    # Defining the optimizer with a learning rate of 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)

    # Evaluate the model on the val set
    loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on val set:", round(binary_accuracy, 7))
    print("Precision on val set:", round(precision, 7))
    print("Recall on val set:", round(recall, 7))

    # Make predictions on the val set
    binary_preds = model.predict(X_val)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_val, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, binary_preds_rounded))
    print("MLP model")

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)

    # Evaluate the model on the val set
    loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
    print("Binary Cross-Entropy Loss:", round(loss, 5))
    print("Binary Accuracy on val set:", round(binary_accuracy, 5))
    print("Precision on val set:", round(precision, 5))
    print("Recall on val set:", round(recall, 5))

    # Make predictions on the val set
    binary_preds = model.predict(X_val)

    # Find the optimal threshold for F1-score and recall trade-off
    best_f1 = 0
    best_threshold = 0.5  # Initial threshold
    for threshold in np.arange(0.15, 0.85, 0.01):
        binary_preds_rounded = [1 if pred > threshold else 0 for pred in binary_preds]
        current_f1 = f1_score(y_val, binary_preds_rounded)
        current_recall = recall_score(y_val, binary_preds_rounded)
        if current_f1 > best_f1 and current_recall > 0.85:  # Adjust the recall threshold as needed
            best_f1 = current_f1
            best_threshold = threshold

    print("Best Threshold for F1-score and Recall Trade-off:", round(best_threshold, 2))
    print("Best F1-score:", round(best_f1, 5))

    # Apply the best threshold to predictions
    binary_preds_rounded_best = [1 if pred > best_threshold else 0 for pred in binary_preds]

    # Calculate Confusion Matrix and F1-score with the best threshold
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, binary_preds_rounded_best))
    print("F1-score with best threshold:", round(f1_score(y_val, binary_preds_rounded_best), 5))
    print("Threshold applied at 85% recall atleast")




if __name__ == '__main__':
    run_program('PyCharm')