# -*- coding:utf-8 -*-
"""
@Code Author   : Gaosong SHI
@Email         : shigs@mail2.sysu.edu.cn
@project       : CSDLv2
@File          : validation.py
@Time          : 2024/11/8 星期五 16:37 
@Software      : PyCharm 
@describe      : 
"""


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

# Load dataset and define variables
df_data = pd.read_csv("../datasets/regressionMatrix/regMatrix_merge.csv")
variable_list = ["OC", "sand", "silt", "clay", "pH", "gravel", "prosity", "cec", "TK", "TN", "TP", "AK", "AN", "AP", "BD"]
china_variables = ["AK", "AN", "AP", "prosity"]

var_match_dict = {
    "cec": "CEC", "gravel": "GRAV", "clay": "CLAY", "pH": "PHH2O", "sand": "SAND", "silt": "SILT",
    "prosity": "POR", "OC": "OC", "TK": "TK", "TN": "TN", "TP": "TP", "AK": "AK", "AN": "AN", "AP": "AP", "BD": "BD"
}

base_path = "/tera05/forShigs/soilChina/otherDataSet/"
layers = {5: (0, 0), 15: (1, 2), 30: (3, 3), 60: (4, 4), 100: (5, 5), 200: (6, -1)}

# Processing function for each variable and layer
def process_variable(variable, lat_cov, lon_cov, cov_data, layer):
    layer_indices = layers[layer]
    mean_data = np.mean(cov_data[layer_indices[0]:layer_indices[1] + 1], axis=0)

    label = df_data[f"{variable}{layer}cm"].values
    lats, lons = df_data["lat"].values, df_data["lon"].values

    lat_indices = [np.abs(lat_cov - lat).argmin() for lat in lats]
    lon_indices = [np.abs(lon_cov - lon).argmin() for lon in lons]
    predicted_data = mean_data[lat_indices, lon_indices]

    combined_data = pd.DataFrame(np.column_stack((predicted_data, label)))
    exclude_values = [-9999, -32768, np.nan, "NaN"]
    selected_data = combined_data[~combined_data.isin(exclude_values).any(axis=1)].dropna()

    return selected_data.iloc[:, 0].values, selected_data.iloc[:, 1].values

# Metric calculation function
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    explained_variance = explained_variance_score(y_true, y_pred)
    return {"R2": r2, "r": r, "RMSE": rmse, "MAE": mae, "ME": me, "Explained Variance": explained_variance}

# Main processing loop
for variable in variable_list:
    dataset_path = "chinaV1/chinaV1Npy" if variable in china_variables else "globalV1/allVarNpy"
    lat_cov = np.load(f"{base_path}/{dataset_path}/lat.npy")
    lon_cov = np.load(f"{base_path}/{dataset_path}/lon.npy")
    cov_data = np.load(f"{base_path}/{dataset_path}/{var_match_dict[variable]}.npy")

    for layer in layers.keys():
        y_pred, y_true = process_variable(variable, lat_cov, lon_cov, cov_data, layer)

        # Scale transformations for specific variables
        if variable in ["BD", "OC", "TK", "TN"]:
            y_pred /= 100
        elif variable == "pH":
            y_pred /= 10
        elif variable == "TP":
            y_pred /= 10000

        metrics = calculate_metrics(y_true, y_pred)
        print(f"\nVariable: {variable}, Layer: {layer}cm")
        print("Metrics:", metrics)
        print("Label Range:", np.min(y_true).round(5), "-", np.max(y_true).round(5))
        print("Prediction Range:", np.min(y_pred).round(5), "-", np.max(y_pred).round(5))

