import pandas as pd
import numpy as np

def rolling_features(df, window=10, functions=['mean', 'std', 'min', 'max'], grad_windows=[3,5]):
    '''
    Generate rolling features for the given DataFrame.
    ''' 

    # Exclude non-numeric columns
    df_feat = df.select_dtypes(include=['number']).copy()

    # for col in df_feat.columns:
    #     df_feat[f"{col}_grad"] = df_feat[col].diff()

    #     # Smoothed gradient using different window sizes
    #     for g in grad_windows:
    #         df_feat[f"{col}_grad_rolling_{g}"] = df_feat[col].diff().rolling(g, min_periods=1).mean()

    # Generate rolling features
    for col in df_feat.columns:
        rolling = df_feat[col].rolling(window=window, min_periods=1)

        for func in functions:
            col_name = f"{col}_rolling_{func}_{window}"
            df_feat[col_name] = getattr(rolling, func)().bfill()

    return df_feat


def augment_minority_continuous_timeseries(X_train, y_train, n_augmentations=3):
    """
    Augment minority class in time-series continuous data by adding Gaussian noise.
    Maintains temporal order without shuffling.
    """
    cols = X_train.columns

    X_minority = X_train[y_train == 1]
    X_majority = X_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    
    # Augment minority
    X_augmented = [X_minority]
    y_augmented = [y_minority]
    
    for _ in range(n_augmentations):
        noise = np.random.normal(0, 0.01 * np.std(X_minority, axis=0), X_minority.shape)
        X_noisy = pd.DataFrame(X_minority.values + noise, columns=cols)
        X_augmented.append(X_noisy)
        y_augmented.append(y_minority)
    
    # Combine majority and augmented minority
    X_balanced = pd.concat([X_majority] + X_augmented, ignore_index=True)
    y_balanced = pd.concat([y_majority] + y_augmented, ignore_index=True)

    return X_balanced, y_balanced


def rolling_slope(series, window):
    """Compute slope of a rolling window using linear regression."""
    idx = np.arange(window)
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series[i-window:i]
            # Simple linear regression slope formula
            slope = np.polyfit(idx, y, 1)[0]
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def build_time_features(df, sensor_cols, 
                        lags=[1, 5, 10], 
                        roll_windows=[10, 30, 60]):
    """
    df: DataFrame with time in correct order
    sensor_cols: list of columns like ['pressure', 'temperature', 'flow']
    lags: list of timesteps to use for lag features
    roll_windows: list of window sizes for rolling statistics
    """

    df = df.copy()
    new_features = {}

    # ---------- RAW FEATURES ----------
    # (we assume df already contains the raw t values)

    # ---------- LAG FEATURES ----------
    print("Generating lag features...")
    for col in sensor_cols:
        for lag in lags:
            new_features[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ---------- ROLLING STATISTICS ----------
    print("Generating rolling statistics features...")
    for col in sensor_cols:
        for w in roll_windows:
            roll = df[col].rolling(window=w)
            new_features[f"{col}_rollmean_{w}"] = roll.mean()
            new_features[f"{col}_rollstd_{w}"]  = roll.std()
            new_features[f"{col}_rollmin_{w}"]  = roll.min()
            new_features[f"{col}_rollmax_{w}"]  = roll.max()
            new_features[f"{col}_rollslope_{w}"] = rolling_slope(df[col], w)
    # ---------- DIFFERENCES ----------
    print("Generating difference features...")
    for col in sensor_cols:
        new_features[f"{col}_diff1"] = df[col].diff(1)
        new_features[f"{col}_diff5"] = df[col].diff(5)
        new_features[f"{col}_diff10"] = df[col].diff(10)

    # ---------- RATIOS ----------
    # Works only if you have at least 2 sensors—modify as needed.
    print("Generating ratio features...")
    if len(sensor_cols) >= 2:
        for i in range(len(sensor_cols)):
            for j in range(i+1, len(sensor_cols)):
                c1, c2 = sensor_cols[i], sensor_cols[j]
                new_features[f"{c1}_to_{c2}_ratio"] = df[c1] / (df[c2] + 1e-6)

    # ---------- COMBINE NEW FEATURES ----------
    print("Combining new features into DataFrame...")
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    # ---------- Drop rows with NaN created by lags/rolling ----------
    print("Dropping rows with NaN values created by lag/rolling operations...")
    df = df.dropna().reset_index(drop=True)

    print("Feature engineering completed.")
    print(f"New number of features: {df.shape[1]}")

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

def pump_pressure_fraction_feature(df, ts_pressure_col="Pressure_Drop", outlet_col="Pump outlet pressure (Mean)"):
    """
    Calculate the fraction of pump outlet pressure to the pressure drop.
    Fraction = ΔP / P_outlet
    """
    df["Pump_Pressure_Ratio"] = df[ts_pressure_col] / (df[outlet_col] + 1e-6) 

    return df



## Flow-pressure interaction feature

def hydraulic_conductance_feature(df, flow_col="Flow rate (Mean)", ts_pressure_col="Pressure_Drop"):
    """
    Calculate hydraulic conductance as a feature.
    C = Q / ΔP
    where Q is flow rate and ΔP is pressure drop.
    """
    df["Hydraulic_Conductance"] = df[flow_col] / (df[ts_pressure_col] + 1e-6)  

    return df


def flow_sensitivity_feature(df, flow_col="Flow rate (Mean)", pump_col="Pump outlet pressure (Mean)"):
    """
    Calculate flow sensitivity to inlet pressure as a feature.
    Sensitivity = dQ / dP_pump
    """
    df["Flow_Sensitivity"] = df[flow_col].diff() / (df[pump_col].diff() + 1e-6)  

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