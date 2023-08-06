# Basic
import pandas as pd
import numpy as np

# Loading, converting, saving libraries
from aws_to_df import AwsToDf
from newtools import PandasDoggo

from aws_to_df import AwsToDf


# Libraries for Plots/Data Visualisations
import matplotlib.pyplot as plt
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

# Performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

# Re-sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

# Tuning
from sklearn.model_selection import GridSearchCV

def run_program(main):
    atd = AwsToDf()
    X_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_train_nw.csv', 'csv', has_header=True)
    X_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_val.csv', 'csv', has_header=True)

    # X_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'X_test.csv', 'csv', has_header=True)

    y_train = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_train_nw.csv', 'csv', has_header=True)

    #y_test = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_test.csv', 'csv', has_header=True)

    y_val = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'y_val.csv', 'csv', has_header=True)

    print("Re-sampled data loaded")
# Dropping Un-named
    X_train = X_train.drop(['Unnamed: 0'], axis=1)
    y_train = y_train.drop(['Unnamed: 0'], axis=1)
    """
    X_test = X_test.drop(['Unnamed: 0'], axis=1)
    y_test = y_test.drop(['Unnamed: 0'], axis=1)
    """
    X_val = X_val.drop(['Unnamed: 0'], axis=1)
    y_val = y_val.drop(["Unnamed: 0"], axis = 1)

    print(X_train.columns)
    # Fine-tuning

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
    print("Best params for XG Boost found")
 
# Create a function to plot learning curves
    def plot_learning_curves(model, X_train, y_train):
        train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1)
       
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

# Plot the learning curves
    plot_learning_curves(xg, X_train, y_train)
    print("Learning curve for XG Boost")
    """
#MLP
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
    """
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
    # Model architecture
    model = tf.keras.Sequential([
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.2f}".format(f1))
    

    # Neurons per layer = 200, number of layers = 4

    tf.random.set_seed(42)
    # Model architecture
    model = tf.keras.Sequential([
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))
    
    # Neurons per layer = 300, number of layers = 4

    tf.random.set_seed(42)
    # Model architecture
    model = tf.keras.Sequential([
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))

# best-3+ optimiser = Adam

    tf.random.set_seed(42)
    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))
    
# Activation fn- tanh

    tf.random.set_seed(42)
    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='tanh', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(200, activation='tanh'),
        tf.keras.layers.Dense(200, activation='tanh'),
        tf.keras.layers.Dense(200, activation='tanh'),
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))

    """
    Computationally heavy---but would lead to better/more assured results for hyper-parameter selection
    import optuna
    # Define the objective function for Optuna
    def objective(trial):
        # Sample hyperparameters to explore
        num_layers = trial.suggest_int('num_layers', 2, 7)  # Suggest a number of layers between 1 and 5

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
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))
    """
    """
# Activation functions- relu + tanh

    tf.random.set_seed(42)

    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(200, activation='tanh'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='tanh'),
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))
    
# Relu + tanh
    tf.random.set_seed(42)

    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='tanh'),
        tf.keras.layers.Dense(200, activation='tanh'),
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))
    """
    tf.random.set_seed(42)

    # Model architecture
    model = tf.keras.Sequential([
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

    # Evaluate the model on the training set
    train_loss, train_binary_accuracy, train_precision, train_recall = model.evaluate(X_train, y_train)
    print("Train Binary Cross-Entropy Loss:", round(train_loss, 7))
    print("Train Binary Accuracy:", round(train_binary_accuracy, 7))
    print("Train Precision:", round(train_precision, 7))
    print("Train Recall:", round(train_recall, 7))

    # Evaluate the model on the val set
    val_loss, val_binary_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val)
    print("Val Binary Cross-Entropy Loss:", round(val_loss, 7))
    print("Val Binary Accuracy:", round(val_binary_accuracy, 7))
    print("Val Precision:", round(val_precision, 7))
    print("Val Recall:", round(val_recall, 7))

    # Evaluate the model on the test set
    test_loss, test_binary_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
    print("Test Binary Cross-Entropy Loss:", round(test_loss, 7))
    print("Test Binary Accuracy:", round(test_binary_accuracy, 7))
    print("Test Precision:", round(test_precision, 7))
    print("Test Recall:", round(test_recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))

    # Detection of model fit given the set hyperparameters
    plt.plot(history.history['binary_accuracy'], label='Training Binary Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Binary Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()
    plt.title('Training Binary Accuracy and Validation Binary Accuracy')
    plt.show()

    print("gjds")

    # Plot the loss curve for training and validation
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    print("lr")


# with regularisation---L1 (Lasso)

    tf.random.set_seed(42)


# with L2 regularisation---Ridge

    tf.random.set_seed(42)

    # Model architecture with L2 regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],),
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
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

    # Evaluate the model on the test set
    loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)
    print("Binary Cross-Entropy Loss:", round(loss, 7))
    print("Binary Accuracy on test set:", round(binary_accuracy, 7))
    print("Precision on test set:", round(precision, 7))
    print("Recall on test set:", round(recall, 7))

    # Make predictions on the test set
    binary_preds = model.predict(X_test)
    # Round the predictions to get the binary class labels (0 or 1)
    binary_preds_rounded = [1 if pred > 0.5 else 0 for pred in binary_preds]

    # Calculate F1-score for binary classification
    f1 = f1_score(y_test, binary_preds_rounded)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, binary_preds_rounded))
    print("F1-score: {:.3f}".format(f1))
    print("F1-score: {:.3f}".format(f1))

if __name__ == '__main__':
    run_program('PyCharm')