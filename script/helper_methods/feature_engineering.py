"""
Helper methods for feature engineering, 
including functions to create derived features.
Also includes a function to calculate a plug index based on pressure drop and flow rate, 
which can be used as an additional feature for plug prediction models.
"""
import numpy as np
import pandas as pd
from script.helper_methods.config import get_config

# Load config
cfg = get_config()
seed = cfg["experiment"]["random_state"]
non_feature_columns = cfg["data"]["non_feature_columns"]

# Derivatives (d/dt) + (d2/dt2)
def add_time_derivative_features(df, time_col="Time"):
    """
    Add first and second time derivative features for all numeric columns in the DataFrame.
    Assumes 'time_col' is in datetime format (e.g., '1900-01-01 11:07:40.450') and sorted.
    """
    # Ensure time column is in datetime format and calculate time differences in seconds
    df[time_col] = pd.to_datetime(df[time_col])
    time_diffs = df[time_col].diff().dt.total_seconds().fillna(1)  
    
    # Get numeric columns excluding non-feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in non_feature_columns]  # Exclude non-feature columns

    # Calculate first and second derivatives for each numeric column
    for col in numeric_cols:
        first_derivative = (df[col].diff() / time_diffs).replace([np.inf, -np.inf], np.nan)
        second_derivative = (first_derivative.diff() / time_diffs).replace([np.inf, -np.inf], np.nan)

        df[f"{col}_d1"] = first_derivative.fillna(0)
        df[f"{col}_d2"] = second_derivative.fillna(0)

    return df


### COMBINED FEATURES ###

def pressure_drop_feature(df, inlet_col="TS inlet pressure (Mean)", outlet_col="TS outlet pressure (Mean)"):
    """
    Calculate pressure drop across the pump as a feature.
    ΔP = P_outlet - P_inlet
    """

    df["TS_in_changed_out"] = 0 # Flag to indicate if inlet column was changed to pump outlet pressure (for cases where TS inlet pressure is not available)

    if inlet_col not in df.columns:
        inlet_col = "Pump outlet pressure (Mean)"
        df["TS_in_changed_out"] = 1


    df["Pressure_Drop"] = df[outlet_col] - df[inlet_col]

    return df

def pump_pressure_fraction_feature(df, ts_pressure_col="Pressure_Drop", outlet_col="Pump outlet pressure (Mean)", clip=1e4):
    """
    Calculate the fraction of pump outlet pressure to the pressure drop.
    Fraction = ΔP / P_outlet
    """
    
    ratio = df[ts_pressure_col] / (df[outlet_col].abs() + 1e-3) 
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    df["Pump_Pressure_Ratio"] = ratio.clip(-clip, clip)

    return df


def feature_engineering_pipeline(df):
    """
    Apply a series of feature engineering steps to the DataFrame.
    """
    df = df.copy()
    
    # Combined features 
    df = pressure_drop_feature(df)
    df = pump_pressure_fraction_feature(df)

    return df

def plug_index(df, window_size=0.5, p_up_col="TS inlet pressure (Mean)", p_down_col="TS outlet pressure (Mean)"):
    """
    Function to calculate Plug Index based on pressure drop.
    Plug index indicates likelihood of plug formation.
    """

    #TODO: remove
    # Add flag to indicate if inlet column was changed to pump outlet pressure (for cases where TS inlet pressure is not available)
    df["TS_in_changed_out"] = 0

    if p_up_col not in df.columns:
        p_up_col = "Pump outlet pressure (Mean)"
        df["TS_in_changed_out"] = 1


    df = df.copy()
    # Check window size
    w = int(window_size / 0.05)
    if w < 2:
        raise ValueError("Window size too small for plug index calculation.")
    
    # Calculate pressure drop and its slope
    df["dP"] = df[p_up_col] - df[p_down_col]
    df["dP_slope"] = df["dP"].rolling(window=w, min_periods=w).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).fillna(0)
    # Standardize dP and dP_slope
    df["dP_z"] = (df["dP"] - df["dP"].mean()) / (df["dP"].std() + 1e-6)
    df["dP_slope_z"] = (df["dP_slope"] - df["dP_slope"].mean()) / (df["dP_slope"].std() + 1e-6)
    
    # Combine standardized pressure drop and slope to create plug index
    df["Plug_Index"] = df["dP_z"] + df["dP_slope_z"] 

    return df