import pandas as pd
import numpy as np


def augment_minority_continuous_timeseries(X, y, n_augmentations=3, noise_frac=0.01, random_state=42):
    rng = np.random.default_rng(random_state)

    X_df = pd.DataFrame(X)  # works whether X is ndarray or DataFrame
    y_ser = pd.Series(y).reset_index(drop=True)

    if y_ser.nunique() < 2:
        return X_df, y_ser

    X_min = X_df[y_ser == 1]
    y_min = y_ser[y_ser == 1]

    # If a fold happens to have no minority, do nothing
    if len(X_min) == 0:
        return X_df, y_ser

    std = X_min.std(axis=0).to_numpy()
    std = np.where(np.isfinite(std), std, 0.0)

    X_aug_parts = [X_df]
    y_aug_parts = [y_ser]

    for _ in range(n_augmentations):
        noise = rng.normal(loc=0.0, scale=noise_frac * std, size=X_min.shape)
        X_noisy = X_min.to_numpy() + noise
        X_aug_parts.append(pd.DataFrame(X_noisy, columns=X_df.columns))
        y_aug_parts.append(y_min)

    X_out = pd.concat(X_aug_parts, ignore_index=True)
    y_out = pd.concat(y_aug_parts, ignore_index=True)

    return X_out, y_out


###  tidsderiventen (d/dt) + andre ordre (d2/dt2) TODO: test this out
def add_time_derivative_features(df, time_col="Time"):
    """
    Add first and second time derivative features for all numeric columns in the DataFrame.
    Assumes 'time_col' is in datetime format (e.g., '1900-01-01 11:07:40.450') and sorted.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    time_diffs = df[time_col].diff().dt.total_seconds().fillna(1)  # Fill NaN for first row

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Plug_future', 'Plug', 'Anomaly']]  # Exclude target columns

    for col in numeric_cols:
        first_derivative = (df[col].diff() / time_diffs).replace([np.inf, -np.inf], np.nan)
        second_derivative = (first_derivative.diff() / time_diffs).replace([np.inf, -np.inf], np.nan)

        df[f"{col}_d1"] = first_derivative.fillna(0)
        df[f"{col}_d2"] = second_derivative.fillna(0)

    return df



### PHYSICS BASED FEATURES ###

def pressure_drop_feature(df, inlet_col="TS inlet pressure (Mean)", outlet_col="TS outlet pressure (Mean)"):
    """
    Calculate pressure drop across the pump as a feature.
    ΔP = P_outlet - P_inlet
    """

    df["TS_in_changed_out"] = 0

    if inlet_col not in df.columns:
        inlet_col = "Pump outlet pressure (Mean)"
        df["TS_in_changed_out"] = 1


    df["Pressure_Drop"] = df[outlet_col] - df[inlet_col]

    return df


def normlized_pressure_drop_feature(df, inlet_col="TS inlet pressure (Mean)", outlet_col="TS outlet pressure (Mean)", flow_col="Flow rate (Mean)"):
    """
    Calculate normalized pressure drop across the pump as a feature.
    Normalized ΔP = (P_outlet - P_inlet) / P_inlet
    """
    df["TS_in_changed_out"] = 0

    if inlet_col not in df.columns:
        inlet_col = "Pump outlet pressure (Mean)"
        df["TS_in_changed_out"] = 1

        
    df["Normalized_Pressure_Drop"] = (df[outlet_col] - df[inlet_col]) / (df[flow_col] + 1e-6)  

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



## Flow-pressure interaction feature

def hydraulic_conductance_feature(df, flow_col="Flow rate (Mean)", ts_pressure_col="Pressure_Drop"): #TODO
    """
    Calculate hydraulic conductance as a feature. 
    C = Q / ΔP
    where Q is flow rate and ΔP is pressure drop.
    """
    df["Hydraulic_Conductance"] = df[flow_col] / (df[ts_pressure_col] + 1e-6)  

    return df


def flow_sensitivity_feature(df, flow_col="Flow rate (Mean)", pump_col="Pump outlet pressure (Mean)"): #TODO
    """
    Calculate flow sensitivity to inlet pressure as a feature.
    Sensitivity = dQ / dP_pump
    """
    df["Flow_Sensitivity"] = (df[flow_col].diff() / (df[pump_col].diff() + 1e-6)).fillna(0)
    return df

### Temperature-based derived features

def ts_temperature_rise(df, inlet_temp_col="Temperature TS inlet (Mean)", outlet_temp_col="Temperature TS outlet (Mean)"):
    """
    Calculate temperature rise across the system as a feature.
    ΔT = T_outlet - T_inlet
    """

    df["Temperature_Rise"] = df[outlet_temp_col] - df[inlet_temp_col]

    return df


def ts_bypass_difference(df, inlet_temp_col="Temperature TS inlet (Mean)", bypass_temp_col="Bypass temperature (Mean)"):
    """
    Calculate temperature difference between TS inlet and bypass as a feature.
    ΔT_bypass = T_inlet - T_bypass
    """
    df["TS_Bypass_Temp_Diff"] = df[inlet_temp_col] - df[bypass_temp_col]

    return df


def feature_engineering_pipeline(df):
    """
    Apply a series of feature engineering steps to the DataFrame.
    """
    df = df.copy()

    # Physics-based features
    df = pressure_drop_feature(df)
    df = normlized_pressure_drop_feature(df)
    df = pump_pressure_fraction_feature(df)
    df = hydraulic_conductance_feature(df)
    df = flow_sensitivity_feature(df)
    df = ts_temperature_rise(df)
    df = ts_bypass_difference(df)

    return df



def plug_index(df, window_size=0.5, p_up_col="TS inlet pressure (Mean)", p_down_col="TS outlet pressure (Mean)", flow_col="Flow rate (Mean)"):
    """
    Function to calculate Plug Index based on pressure drop and flow rate.
    Plug index indicates likelihood of plug formation.
    """

    df["TS_in_changed_out"] = 0

    if p_up_col not in df.columns:
        p_up_col = "Pump outlet pressure (Mean)"
        df["TS_in_changed_out"] = 1

    df = df.copy()

    w = int(window_size / 0.05)
    if w < 2:
        raise ValueError("Window size too small for plug index calculation.")
    
    df["dP"] = df[p_up_col] - df[p_down_col]

    df["dP_slope"] = df["dP"].rolling(window=w, min_periods=w).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).fillna(0)

    df["dP_z"] = (df["dP"] - df["dP"].mean()) / (df["dP"].std() + 1e-6)
    df["dP_slope_z"] = (df["dP_slope"] - df["dP_slope"].mean()) / (df["dP_slope"].std() + 1e-6)
    df["flow_z"] = (df[flow_col] - df[flow_col].mean()) / (df[flow_col].std() + 1e-6)

    df["Plug_Index"] = df["dP_z"] + df["dP_slope_z"] - df["flow_z"]

    return df