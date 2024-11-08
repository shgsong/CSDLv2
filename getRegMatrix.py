# -*- coding:utf-8 -*-
"""
@Code Author   : Gaosong SHI
@Email         : shgsong@foxmail.com
@project       : CSDLv2
@File          : getRegMatrix.py
@Time          : 2023/11/22 0022 22:25
@Software      : PyCharm
@describe      : TODO
"""

import os
import numpy as np
import pandas as pd
import datetime
import warnings

warnings.filterwarnings("ignore")

# Define paths and constants
MEMMAP_PATH = "../datasets/covs/chinaCovsMemmap/"
SOIL_PROFILE_PATH = "../datasets/soilProfiles/"
OUTPUT_FOLDER_PATH = "../datasets/regressionMatrix/"

VARIABLE_LIST = ["OC", "sand", "silt", "clay", "pH", "gravel", "prosity", "cec", "TK", "TN",
                 "TP", "AK", "AN", "AP", "BD", "Wet_R", "Wet_G", "Wet_B", "Dry_R", "Dry_G", "Dry_B"]
LAYER_DEPTHS = [5, 15, 30, 60, 100, 200]

# Track the start time of the program
start_time = datetime.datetime.now()


# Load covariate data
def load_covariate_data():
    memmap_data = np.memmap(
        os.path.join(MEMMAP_PATH, "chinaCovsMemmap.npy"),
        dtype=np.float32,
        mode="r",
        shape=(264, 43000, 74011)  # Ensure the correct shape
    )
    feature_names = np.load(os.path.join(MEMMAP_PATH, "chinaCovsName.npy"))
    latitude_arr = np.load(os.path.join(MEMMAP_PATH, "latitude.npy"))
    longitude_arr = np.load(os.path.join(MEMMAP_PATH, "longitude.npy"))
    print("Number of covariates:", len(feature_names))
    print("Latitude array length:", len(latitude_arr))
    print("Longitude array length:", len(longitude_arr))
    return memmap_data, feature_names, latitude_arr, longitude_arr


# Load soil profile data
def load_soil_profile_data(filename="soilPoint.csv"):
    soil_profile_path = os.path.join(SOIL_PROFILE_PATH, filename)
    return pd.read_csv(soil_profile_path, encoding="GB18030")


# Retrieve covariate values for profiles
def get_covariates_for_profiles(memmap_data, latitude_arr, longitude_arr, df_soil_profile):
    total_covariates = []
    for lat, lon in zip(df_soil_profile["lat"].values, df_soil_profile["lon"].values):
        lat_idx = np.abs(latitude_arr - lat).argmin()
        lon_idx = np.abs(longitude_arr - lon).argmin()
        covariates = memmap_data[:, lat_idx, lon_idx]
        total_covariates.append(covariates)
    return np.array(total_covariates)


# Load and integrate label data
def load_labels(df_soil_profile, var_name, layer):
    labels_path = os.path.join(f"../datasets/spline/std_{var_name}.csv")
    df_labels = pd.read_csv(labels_path)
    labels_list = []
    for code in df_soil_profile["Code"]:
        label = df_labels[(df_labels["no"] == code) & (df_labels['depth'] == layer)][var_name].tolist()
        labels_list.append(label[0] if label and label[0] > 0 else np.nan)
    return labels_list


# Main function to integrate covariates and labels
def main():
    memmap_data, feature_names, latitude_arr, longitude_arr = load_covariate_data()
    df_soil_profile = load_soil_profile_data()

    # Get covariate values for each profile
    total_covariates_values = get_covariates_for_profiles(memmap_data, latitude_arr, longitude_arr, df_soil_profile)
    df_covariates = pd.DataFrame(total_covariates_values, columns=feature_names)

    # Initialize output DataFrame
    df_output = pd.DataFrame({
        "Code": df_soil_profile["ID"],
        "lat": df_soil_profile["latitude"],
        "lon": df_soil_profile["longitude"],
    })
    df_output = pd.concat([df_output, df_covariates], axis=1)

    # Add labels for each layer depth
    for layer in LAYER_DEPTHS:
        for var_name in VARIABLE_LIST:
            print(f"Loading depth {layer}cm and variable {var_name}")
            labels = load_labels(df_soil_profile, var_name, layer)
            df_output[f"{var_name}{layer}cm"] = labels

    # Save to CSV file
    output_path = os.path.join(OUTPUT_FOLDER_PATH, "regMatrix.csv")
    df_output.to_csv(output_path, index=False)
    print("Output file shape:", df_output.shape)
    print("Processing time:", datetime.datetime.now() - start_time)


# Run main function
if __name__ == "__main__":
    main()





