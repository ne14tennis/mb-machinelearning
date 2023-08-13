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
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
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
    # Chi-squared 300 feature names
    new_names = [ 'is_latest_season', 'is_national', 'hour_of_day', '376', '377', '378',
    '386', '493', '585', '649', '2382', '2383', '2384', '3630', '3633',
    '3637', '3645', '3646', '3656', '5038', '5043', '5070', '10946',
    '53749', '53750', '53751', '64893', '79929', '79931', '79934', 'Friday',
    'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
    'h 100172451', 'h 100318334', 'h 106306721', 'h 106308521',
    'h 107043460', 'h 107434320', 'h 109046304', 'h 110590951',
    'h 110788150', 'h 111285788', 'h 112227013', 'h 115565015',
    'h 115742946',
    'h 115837803', 'h 115885426', 'h 116004824', 'h 116488929',
    'h 125514781', 'h 125807126', 'h 165898931', 'h 167160672',
    'h 167514439', 'h 167839487', 'h 167917001', 'h 170268898',
    'h 172083888', 'h 173055791', 'h 2632155', 'h 35928955', 'h 36074620',
    'h 36280817', 'h 36648962', 'h 36683815', 'h 36799727', 'h 37322640',
    'h 37776317', 'h 37890042', 'h 38051546', 'h 39292432', 'h 46572705',
    'h 46842762', 'h 47101504', 'h 47650429', 'h 47830694', 'h 47878021',
    'h 48636316', 'h 48842286', 'h 48891969', 'h 48931642', 'h 48946919',
    'h 48999040', 'h 49160365', 'h 49339329', 'h 49447347', 'h 49616626',
    'h 49791131', 'h 50768362', 'h 50905501', 'h 50918177', 'h 51674347',
    'h 51674918', 'h 51741527', 'h 52146982',
    'h 52388485', 'h 53963599', 'h 54023407', 'h 54120731', 'h 54717301',
     'h 55494075', 'h 55500128', 'h 55539553', 'h 55814419', 'h 56008797',
     'h 56050804', 'h 56135757', 'h 56360685', 'h 56537778', 'h 56693912',
     'h 56853304', 'h 56885748', 'h 56924259', 'h 57274298', 'h 57480793',
     'h 57579254', 'h 57592566', 'h 57600223', 'h 57622186', 'h 57643519',
     'h 57741341', 'h 57749081', 'h 58277309', 'h 58717993', 'h 59031388',
     'h 59109678', 'h 59114829', 'h 59319212', 'h 59881658', 'h 59988152',
     'h 60243206', 'h 60641874', 'h 60669224', 'h 60823025', 'h 60844917',
     'h 61312789', 'h 61333557', 'h 61554804', 'h 61637075', 'h 62561554',
     'h 62890430', 'h 63142558', 'h 63270041', 'h 64448593', 'h 64547438',
    'h 64651548', 'h 64670207', 'h 64795412', 'h 64920613', 'h 65338118',
     'h 65601007', 'h 65909820', 'h 66206556', 'h 66255849', 'h 66425354',
     'h 66434011', 'h 67243422', 'h 67568449', 'h 67757110', 'h 67927920',
     'h 67982228', 'h 68221055', 'h 68489793', 'h 68512332', 'h 68998524',
     'h 69292040', 'h 69988471', 'h 69989997', 'h 70003848', 'h 70025619',
     'h 70051375', 'h 70081131', 'h 70084286', 'h 70258850', 'h 70553680',
     'h 70740042', 'h 70783562', 'h 70808979', 'h 70867299', 'h 71162977',
     'h 72134555', 'h 72226381', 'h 72434949', 'h 72516526', 'h 72574311',
     'h 72631321', 'h 72700283', 'h 73399516', 'h 73874081', 'h 73898027',
     'h 75234809', 'h 75571504', 'h 75930190', 'h 76012513', 'h 77153099',
     'h 78535346', 'h 78734284', 'h 79550177', 'h 79980408', 'h 80339662',
     'h 81371160', 'h 85016297', 'h 90776857', 'h 91297905', 'h 91493515',
     'h 91509340', 'h 91908121', 'h 91958986', 'h 92643920', 'h 93079773',
     'h 93583160', 'h 93837110', 'h 97212190', 'h 97672913', 'h 98700821',
     'h 98931594', 'h 99358595', 'n 33', 'n 4', 'n 5', 'st -51039',
     'st -57526', 'st 10079', 'st 10695', 'st 12340', 'st 12770', 'st 13496',
     'st 19462', 'st 26078', 'st 26089', 'st 26912', 'st 28922', 'st 4203',
     'st 42032', 'st 4207', 'st 4343', 'st 4491', 'st 4576', 'st 5156',
     'st 5209', 'st 5289', 'st 5314', 'st 5356', 'st 5393', 'st 5554',
    'st 5652', 'st 5739', 'st 6421', 'st 6704', 'st 9108', 'st 9532',
     'st 9540', '1.0', '2.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
     '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '18.0', '19.0',
     '20.0', '21.0', '28.0', '54.0', 'g 10', 'g 11', 'g 13', 'g 17', 'g 18',
     'g 19', 'g 20', 'g 23', 'g 24', 'g 25', 'g 26', 'g 4', 'g 5', 'g 6',
     'g 7', 'g 9', 'mk 2204', 'mk 2205', 'mk 2206', 'mk 2207', 'mk 2208',
     'mk 2209']



     # Rename columns for X_train
    X_train.columns = new_names

    # Rename columns for X_test
    X_test.columns = new_names

    # Rename columns for X_val
    X_val.columns = new_names


    print("Break")
#MLP model
    """
    # Neurons per layer = 100
    tf.random.set_seed(42)
    np.random.seed(42)

    # Define normalization layer
    norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])

    # Model architecture
    model = tf.keras.Sequential([
        norm_layer,
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Defining the optimizer with a learning rate of 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Compiling the model with binary cross-entropy loss and metrics
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=36,
                        validation_data=(X_val, y_val),
                        verbose=2)

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
    print("F1-score: {:.5f}".format(f1))
    print("F1-score: {:.5f}".format(f1))
    """

# XG Boost
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
    print("XG Boost")

    # Calculate Feature Importance Scores
    feature_importance = xg.feature_importances_

    # Sort and get the indices of the top 20 features
    sorted_idx = np.argsort(feature_importance)[::-1][:20]

    # Get the names of the top 20 features
    top_feature_names = X_train.columns[sorted_idx]

    # Get the corresponding feature importance scores for the top 20 features
    top_feature_importance = feature_importance[sorted_idx]

    # Plot the top 20 features
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_feature_names)), top_feature_importance, align="center")
    plt.yticks(range(len(top_feature_names)), top_feature_names)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 20 Features by Importance")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top
    plt.show()


    print('Feature importance')

    # Calculate Precision-Recall curve on test set
    precision_test, recall_test, _ = precision_recall_curve(y_test, xg_y_pred)
    # Plot Precision-Recall curve on test set
    plt.subplot(1, 2, 2)
    plt.plot(recall_test, precision_test, color='blue', lw=2,
             label='Precision-Recall curve (area = %0.2f)' % auc(recall_test, precision_test))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
    print("PR Curve")
# Learning Curve

    def plot_comp_in(xg, X_train, y_train):
        train_sizes, train_scores, val_scores = learning_curve(xg, X_train, y_train, cv=5,
                                                               train_sizes=np.linspace(0.1, 1.0, 5),
                                                               scoring=make_scorer(f1_score),
                                                               n_jobs=-1)

        # Calculate the mean and standard deviation for training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot the curve to gauge dataset training requirement
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training F1 Score', color='blue')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, label='Validation F1 Score', color='orange')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    plot_comp_in(xg, X_train, y_train)
    print("Comp requirements check")

if __name__ == '__main__':
    run_program('PyCharm')