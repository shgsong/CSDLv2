# -*- coding:utf-8 -*-
"""
@Code Author   : Gaosong SHI
@Email         : shigs@mail2.sysu.edu.cn
@File          : uncertainty.py
@Time          : 2024/03/19
@Software      : PyCharm
@describe      : TODO

"""
import ast
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor
# from regression_metrics import picp

# Suppress warnings
warnings.filterwarnings("ignore")

# List of variables and layers
variable_list = ["OC", "sand", "silt", "clay", "pH", "gravel", "prosity", "cec", "TK", "TN",
                 "TP", "AK", "AN", "AP", "BD", "W_R", "W_G", "W_B", "D_R", "D_G", "D_B"]
layers = [5, 15, 30, 60, 100, 200]

def load_best_params(file_path):
    """Load and return the best parameters from a CSV file."""
    csv_best_params = pd.read_csv(file_path).sort_values(by="RMSE", ascending=True)
    return ast.literal_eval(csv_best_params.iloc[0, 0])

def load_selected_features(file_path):
    """Load and return the selected features from a CSV file."""
    return list(pd.read_csv(file_path).iloc[:, 0])

def train_and_evaluate_model(variable, layer):
    """Train and evaluate the model for a given variable and layer."""
    data_path = "../datasets/regressionMatrix_csv/regMatrix_merge.csv"
    df_data = pd.read_csv(data_path)  # Load merged dataset

    best_params_path = f'../others/Grid_Search/{variable}_{layer}cm_grid_search_parameters_results.csv'
    selected_features_path = f'../others/Grid_Search/{variable}_{layer}cm_selected_features_results.csv'

    # Load best parameters and selected features
    best_params = load_best_params(best_params_path)
    selected_features_list = load_selected_features(selected_features_path)

    # Prepare the data
    X = df_data[selected_features_list]
    y = df_data[f"{variable}{layer}cm"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize and train the model
    qrf = RandomForestQuantileRegressor(oob_score=True, n_jobs=-1, random_state=42, **best_params)
    qrf.fit(X_train, y_train)

    # Get predictions at 90% prediction intervals and median
    y_pred_qrf = qrf.predict(X_test, quantiles=[0.05, 0.5, 0.95])
    y_pred_lower, y_pred_upper = y_pred_qrf[:, 0], y_pred_qrf[:, 2]

    # Calculate PICP score
    score_picp = picp(y_test, y_pred_lower, y_pred_upper)

    # Print PICP score
    print(f"Variable: {variable}, Layer: {layer}cm, PICP: {score_picp}")

    # Save the trained model
    model_file = f'../models/QRF_model_pkl/{variable}_{layer}cm.pkl'
    joblib.dump(qrf, model_file)




# Main loop to process each variable and layer
for variable in variable_list:
    for layer in layers:
        train_and_evaluate_model(variable, layer)





