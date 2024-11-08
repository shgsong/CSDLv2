# -*- coding:utf-8 -*-
"""
@Code Author   : Gaosong SHI
@Email         : shigs@mail2.sysu.edu.cn
@File          : modeling.py
@Software      : PyCharm
@describe      : TODO

"""

import ast
import joblib
import warnings
import numpy as np
import pandas as pd
from regression_metrics import mean_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# Suppress warnings
warnings.filterwarnings("ignore")

# List of variables and layers
variables = ["OC", "sand", "silt", "clay", "pH", "gravel", "prosity", "cec", "TK", "TN",
             "TP", "AK", "AN", "AP", "BD", "Wet_R", "Wet_G", "Wet_B", "Dry_R", "Dry_G", "Dry_B"]
layers = [5, 15, 30, 60, 100, 200]

def load_best_params(file_path):
    """Load and return the best parameters from a CSV file."""
    best_params_df = pd.read_csv(file_path).sort_values(by="RMSE", ascending=True)
    return ast.literal_eval(best_params_df.iloc[0, 0])

def load_selected_features(file_path):
    """Load and return the selected features from a CSV file."""
    return list(pd.read_csv(file_path).iloc[:, 0])

def train_and_evaluate_model(variable, layer):
    """Train and evaluate the model for a given variable and layer."""
    # Load the dataset
    data_path = "../datasets/regressionMatrix/regMatrix_merge.csv"
    df = pd.read_csv(data_path)

    # Define paths to parameter and feature files
    best_params_file = f'../others/Grid_Search/{variable}_{layer}cm_grid_search_parameters_results.csv'
    selected_features_file = f'../others/Grid_Search/{variable}_{layer}cm_selected_features_results.csv'

    # Load best parameters and selected features
    best_params = load_best_params(best_params_file)
    selected_features = load_selected_features(selected_features_file)

    # Prepare the data by splitting into train and test sets based on testId
    testId = np.load(r"../datasets/testProfiles.npy")
    trainData = df[~df["Code"].isin(testId)]
    testData = df[df["Code"].isin(testId)]

    # Define training and testing features and target variables
    X_train, y_train = trainData[selected_features], trainData[f"{variable}{layer}cm"]
    X_test, y_test = testData[selected_features], testData[f"{variable}{layer}cm"]

    # Initialize and train the model with the best parameters
    model = RandomForestRegressor(oob_score=True, n_jobs=-1, **best_params)

    # Perform 10-fold cross-validation on the training set
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    # Train the model on the full training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mec = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    me = mean_error(y_test, y_pred)

    # Save the trained model to a file
    model_file = f'../models/model_pkl/{variable}_{layer}cm.pkl'
    joblib.dump(model, model_file)

    # Print metrics for the current variable and layer
    print(f"Variable: {variable}, Layer: {layer}cm")
    print("mecScore:", mec)
    print("rmse:", rmse)
    print("ME:", me)
    print("-" * 40)

# Main loop to process each variable and layer
for variable in variables:
    for layer in layers:
        train_and_evaluate_model(variable, layer)



