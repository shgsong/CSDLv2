# -*- coding:utf-8 -*-
"""
@Code Author   : Gaosong SHI
@Email         : shigs@mail2.sysu.edu.cn
@File          : modeling.py
@Time          : 2024/03/19
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

# Suppress warnings
warnings.filterwarnings("ignore")


# List of variables and layers
variables = ["OC", "sand", "silt", "clay", "pH", "gravel", "prosity", "cec", "TK", "TN",
             "TP", "AK", "AN", "AP", "BD", "W_R", "W_G", "W_B", "D_R", "D_G", "D_B"]
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
    data_path = "../datasets/regressionMatrix_csv/regMatrix_merge.csv"
    df = pd.read_csv(data_path)

    best_params_file = f'../others/Grid_Search/{variable}_{layer}cm_grid_search_parameters_results.csv'
    selected_features_file = f'../others/Grid_Search/{variable}_{layer}cm_selected_features_results.csv'

    # Load best parameters and selected features
    best_params = load_best_params(best_params_file)
    selected_features = load_selected_features(selected_features_file)

    # Prepare the data
    X = df[selected_features]
    y = df[f"{variable}{layer}cm"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(oob_score=True, n_jobs=-1, **best_params)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mec = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    me = mean_error(y_test, y_pred)

    # Save the trained model
    model_file = f'../models/model_pkl/{variable}_{layer}cm.pkl'
    joblib.dump(model, model_file)

    # Print metrics
    print(f"Variable: {variable}, Layer: {layer}cm")
    print("mecScore:", mec)
    print("rmse:", rmse)
    print("ME:", me)
    print("-" * 40)





# Main loop to process each variable and layer
for variable in variables:
    for layer in layers:
        train_and_evaluate_model(variable, layer)



