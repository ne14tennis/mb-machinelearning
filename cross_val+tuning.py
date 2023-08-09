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

class a3:
    def sel_tun(self):
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
        y_val = y_val.drop(["Unnamed: 0"], axis=1)

        print(X_train.columns)
        # K-fold Cross Validation

        # MLP
        tf.random.set_seed(42)
        np.random.seed(42)

        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compiling the model with binary cross-entropy loss and metrics
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.BinaryCrossentropy()])

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=36, validation_data=(X_val, y_val),
                            verbose=2)
        print("MLP model defined")

        # List of models
        def get_models():
            models = dict()
            models['log'] = LogisticRegression(random_state=77, max_iter=1000)
            models['dt'] = DecisionTreeClassifier(random_state=42)
            models['rfc'] = RandomForestClassifier(n_estimators=100, random_state=42)
            models['xgb'] = XGBClassifier()
            return models

        # evaluate the Keras model using F1 score
        def evaluate_keras_model(model, X, Y):
            y_pred = model.predict(X)
            y_pred = (y_pred > 0.5).astype(int)
            f1 = f1_score(Y, y_pred)
            return f1

        # Evaluate using cross-validation
        def evaluate_model(model, X, Y, metric='f1'):
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
            scores = cross_val_score(model, X, Y, scoring=metric, cv=cv, n_jobs=-1, error_score='raise')
            return scores

        # get the models to evaluate
        models = get_models()

        # Evaluate the Keras model
        keras_f1 = evaluate_keras_model(model, X_train, y_train)

        # Evaluate the scikit-learn models and store results
        results, names = list(), list()
        metric = 'f1'
        for name, model in models.items():
            scores = evaluate_model(model, X_train, y_train, metric=metric)
            results.append(scores)
            names.append(name)

        # Add the Keras model's F1 score to the results list and names list
        results.append([keras_f1])
        names.append('MLP')

        # Plot model performance for comparison
        plt.boxplot(results, labels=names, showmeans=True)
        plt.title('Model Performance Comparison')
        plt.ylabel(f'{metric.upper()} Score')
        plt.show()
        print("Cross val")

        print("Cross-val")

        # Fine-tuning

        # XG Boost
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
    # Best XG Boost
    
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
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Best params for xg boost")
    
        # Function to plot LC
    
        def plot_lc_n_estimators_f1(xg, X_train, y_train, n_estimators_list):
            f1_scorer = make_scorer(f1_score)
            train_scores, val_scores = validation_curve(xg, X_train, y_train, param_name='n_estimators',
                                                        param_range=n_estimators_list, cv=5,
                                                        scoring=f1_scorer, n_jobs=-1)
    
            # Calculate the mean and standard deviation for training and validation scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
    
            # Plot the learning curves
            plt.figure(figsize=(10, 6))
            plt.plot(n_estimators_list, train_mean, label='Training F1 Score', color='dodgerblue')
            plt.fill_between(n_estimators_list, train_mean - train_std, train_mean + train_std, alpha=0.1, color='dodgerblue')
            plt.plot(n_estimators_list, val_mean, label='Validation F1 Score', color='orange')
            plt.fill_between(n_estimators_list, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
            plt.xlabel('Number of Estimators')
            plt.ylabel('F1 Score')
            plt.title('Learning Curves over Number of Estimators (F1 Score)')
            plt.legend(loc='best')
            plt.grid()
            plt.show()
    
        # Define a list of n_estimators values to plot
        n_estimators_list = [50, 100, 200, 300, 400, 500, 600, 700, 750]
    
        # Plotting LC over n_estimators using F1 score
        plot_lc_n_estimators_f1(xg, X_train, y_train, n_estimators_list)
        print("LC Curve for XG boost")
    
        def plot_comp_in(xg, X_train, y_train):
            train_sizes, train_scores, val_scores = learning_curve(xg, X_train, y_train, cv=5,
                                                                   train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy',
                                                                   n_jobs=-1)
    
            # Calculate the mean and standard deviation for training and validation scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
    
            # Plot the curve to gauge dataset training requirement (Business imp)
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            plt.plot(train_sizes, val_mean, label='Validation Accuracy', color='orange')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy')
            plt.title('Learning Curves')
            plt.legend(loc='best')
            plt.grid()
            plt.show()
    
        # Plotting
        plot_comp_in(xg, X_train, y_train)
    
    
        # L2 reg attempted with lambda=0.01
    
        xg_r = XGBClassifier(random_state=77, colsample_bytree=0.3, learning_rate=0.3, max_depth=6,
                           n_estimators=750, reg_lambda=0.01)
    
        xg_r.fit(X_train, y_train)
    
        # Prediction on test set
        xg_y_pred = xg_r.predict(X_test)
    
        # Prediction Performance Metrics
        accuracy = accuracy_score(y_test, xg_y_pred)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, xg_y_pred)
        precision = precision_score(y_test, xg_y_pred)
        recall = recall_score(y_test, xg_y_pred)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, xg_y_pred))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("L2 regularisation attempted")
        # Plot regularised LC
        plot_l_c(xg, X_train, y_train)
        print("Learning Curve for XG Boost")
        """
        # MLP
        """
        params considered-
        flatten/ normal input layer
        activation functions: ReLu, tanh
        output activation function: sigmoid
        no. of neurons: uniform, pyrimad---10, (10,6,4), 16, 32, 90, 100, 200
        no. of layers: 1,2,3,4,10
        optimiser: sgd, Adam
        loss fn : binary cross entropy
        LR: 0.01, 
        reqularisation- with/without L1, L2
        data processing hyper-parameters [epochs: 50,100,200
                        batch size:    ]
        """
        # Base model
        """
        tf.random.set_seed(42)
        np.random.seed(42)
        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on test set:", round(binary_accuracy,7))
        print("Precision on test set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    
    # Normalisation layer
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=50, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on test set:", round(binary_accuracy,7))
        print("Precision on test set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # Batch Size = 64
    
        tf.random.set_seed(42)
        np.random.seed(42)
        # Model architecture
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on test set:", round(binary_accuracy,7))
        print("Precision on test set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.4f}".format(f1))
        print("F1-score: {:.4f}".format(f1))
    
    # Batch size = 16
        tf.random.set_seed(42)
        np.random.seed(42)
        # Model architecture
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on test set:", round(binary_accuracy,7))
        print("Precision on test set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.4f}".format(f1))
        print("F1-score: {:.4f}".format(f1))
    #EPOCH = 100
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy :", round(binary_accuracy,7))
        print("Precision on val set:", round(precision,7))
        print("Recall on val set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # EPOCH =32
        tf.random.set_seed(42)
        np.random.seed(42)
        # Model architecture
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=32, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on test set:", round(binary_accuracy,7))
        print("Precision on test set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # EPOCH = 100, batch size = 64
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the test set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on test set:", round(binary_accuracy,7))
        print("Precision on test set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    # Epochs = 300 with early stopping where patience = 10
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
        # Defining the optimizer with a learning rate of 0.01
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])
    
        # Define early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    
        # Train the model
        history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping], verbose=2)
    
        # Evaluate the model on the test set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy :", round(binary_accuracy,7))
        print("Precision :", round(precision,7))
        print("Recall on val set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    #LR: 0.1
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
        # Defining the optimizer with a learning rate of 0.1
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    
        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])
    
        # Train the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on val set:", round(binary_accuracy,7))
        print("Precision on val set:", round(precision,7))
        print("Recall on test set:", round(recall,7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # lr: 0.001
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
        # Defining the optimizer with a learning rate of 0.1
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    
        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])
    
        # Train the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on val set:", round(binary_accuracy,7))
        print("Precision on val set:", round(precision,7))
        print("Recall on val set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
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
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy on val set:", round(binary_accuracy,7))
        print("Precision on val set:", round(precision,7))
        print("Recall on val set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # Neurons per layer = 200
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(200, activation='relu'),
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    #Neurons per layer = 10
        # Neurons per layer = 100
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(10, activation='relu'),
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # number of layers = 3/ 1 dense layer
        # Neurons per layer = 100
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    
    #number of layers =10/ 8 dense layers
        tf.random.set_seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
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
        history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy on val set:", round(binary_accuracy, 7))
        print("Precision on val set:", round(precision, 7))
        print("Recall on val set:", round(recall, 7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
        # Neurons per layer = 100, number of layers = 4
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
        # Neurons per layer = 200, number of layers = 4
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
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
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy on val set:", round(binary_accuracy, 7))
        print("Precision on val set:", round(precision, 7))
        print("Recall on val set:", round(recall, 7))
    
        # Make predictions on the test set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
        # Neurons per layer = 300, number of layers = 4
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(300, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(300, activation='relu'),
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # Neurons = pyrimad, layers =4
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # best+ hidden layers = 3
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='relu'),
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
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy :", round(binary_accuracy, 7))
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # best-3+ optimiser = Adam
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
        # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])
    
        # Train the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=2)
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy :", round(binary_accuracy, 7))
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
    # Activation fn- tanh
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='tanh', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='tanh'),
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
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy :", round(binary_accuracy, 7))
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
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
    
        """
        """
        Computationally heavy---but would lead to better/more assured results for hyper-parameter selection
        import optuna
        # Define the objective function for Optuna
        def objective(trial):
            # Sample hyperparameters to explore
            num_layers = trial.suggest_int('num_layers', 2, 7)  
    
            # Model architecture with the suggested number of layers and neurons
            model = tf.keras.Sequential()
    
            # Number of neurons in each layer will be a hyperparameter for each layer
            for i in range(num_layers):
                num_neurons = trial.suggest_int(f'num_neurons_layer{i + 1}', 50, 300)
                model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
    
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
            # Compiling the model with binary cross-entropy loss and Binary_Accuracy as the metric
            model.compile(loss='binary_crossentropy', optimizer='sgd',
                          metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall()])
    
            # Train the model
            model.fit(X_train, y_train, epochs=100, batch_size=36, validation_data=(X_val, y_val), verbose=0)
    
            # Evaluate the model on the validation set
            loss, _, _, _ = model.evaluate(X_val, y_val)
    
            # Return the validation loss as the objective value to minimize
            return loss
    
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
    
        # Access the best hyperparameters found by Optuna
        best_params = study.best_params
    
        # Create and compile the best model with the best hyperparameters
        best_model = tf.keras.Sequential()
    
        for i in range(best_params['num_layers']):
            num_neurons = best_params[f'num_neurons_layer{i + 1}']
            best_model.add(tf.keras.layers.Dense(num_neurons, activation='tanh'))
    
        best_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
        best_model.compile(loss='binary_crossentropy', optimizer='sgd',
                           metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall()])
    
        # Train the best model on the entire training set
        best_model.fit(X_train, y_train, epochs=100, batch_size=36, verbose=2)
    
        # Evaluate the best model on the test set
        loss, binary_accuracy, precision, recall = best_model.evaluate(X_test, y_test)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy on test set:", round(binary_accuracy, 7))
        print("Precision on test set:", round(precision, 7))
        print("Recall on test set:", round(recall, 7))
    
        # Make predictions on the test set using the best model
        binary_preds = best_model.predict(X_test)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification using the best model
        f1 = f1_score(y_test, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))
        """

        """
    # Activation functions- relu + tanh
    
        tf.random.set_seed(42)
        np.random.seed(42)
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='tanh'),
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
    
        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss,7))
        print("Binary Accuracy :", round(binary_accuracy,7))
        print("Precision on val set:", round(precision,7))
        print("Recall on val set:", round(recall,7))
    
        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]
    
        # Calculate F1-score for binary classification
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1
        """
        # With L2 regularisation---Ridge with 0.001
        tf.random.set_seed(42)
        np.random.seed(42)

        # Regularization strength
        l2_reg_strength = 0.001

        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
        # Model architecture
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg_strength),
                                  input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg_strength)),
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

        # Evaluate the model on the val set
        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy:", round(binary_accuracy, 7))
        print("Precision on val set:", round(precision, 7))
        print("Recall on val set:", round(recall, 7))

        # Make predictions on the val set
        binary_preds = model.predict(X_val)
        # Round the predictions to get the binary class labels (0 or 1)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        # Calculate F1-score and confusion matrix
        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("F1-score: {:.5f}".format(f1))

        # Best
        # Convert labels to float32
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Define normalization layer
        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])

        # Model architecture
        mlp_best = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Define optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Compile the model
        mlp_best.compile(loss='binary_crossentropy', optimizer=optimizer,
                         metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                                  tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])

        # Define number of epochs
        epochs = 100

        # Create lists to store F1 scores for training and validation sets
        train_f1_scores = []
        val_f1_scores = []
        # Train the model
        for epoch in range(epochs):
            history = mlp_best.fit(X_train, y_train, epochs=1, batch_size=36, validation_data=(X_val, y_val), verbose=2)

            # Calculate F1 scores for training set
            train_preds = mlp_best.predict(X_train)
            train_preds_rounded = [1 if pred > 0.5 else 0 for pred in train_preds]
            train_f1 = f1_score(y_train, train_preds_rounded)
            train_f1_scores.append(train_f1)

            # Calculate F1 scores for validation set
            val_preds = mlp_best.predict(X_val)
            val_preds_rounded = [1 if pred > 0.5 else 0 for pred in val_preds]
            val_f1 = f1_score(y_val, val_preds_rounded)
            val_f1_scores.append(val_f1)

        # Learning Curve
        plt.plot(range(1, epochs + 1), train_f1_scores, label='Training F1 Score')
        plt.plot(range(1, epochs + 1), val_f1_scores, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Training and Validation F1 Scores')
        plt.show()

        print("Check")
        # Detection of model fit given the set hyperparameters----CALLING FN

        print("Graph")

        # Early Stopping

        tf.random.set_seed(42)
        np.random.seed(42)

        norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])

        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        # Define early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        # Train the model with early stopping
        history = model.fit(X_train, y_train, epochs=100, batch_size=36,
                            validation_data=(X_val, y_val), verbose=2, callbacks=[early_stopping])

        loss, binary_accuracy, precision, recall = model.evaluate(X_val, y_val)
        print("Binary Cross-Entropy Loss:", round(loss, 7))
        print("Binary Accuracy:", round(binary_accuracy, 7))
        print("Precision on val set:", round(precision, 7))
        print("Recall on val set:", round(recall, 7))

        binary_preds = model.predict(X_val)
        binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

        f1 = f1_score(y_val, binary_preds_rounded)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, binary_preds_rounded))
        print("F1-score: {:.5f}".format(f1))
        print("Early stopping implemented")


