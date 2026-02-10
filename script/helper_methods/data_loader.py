import pandas as pd
import glob

def read_unify_data(path="../data/raw_data/data1/*.csv"):
    '''Read a CSV file with unified separator and decimal'''

    with open(path, 'r') as file:
        heaeder_line = file.readline()

    sep = ';' if ';' in heaeder_line else ','

    decimal = '.'

    with open(path, 'r') as file:
        for _ in range(5):
            line = file.readline()
            if ',' in line and sep == ';':
                decimal = ','
                break
            if '.' in line and sep == ',':
                decimal = '.'
                break
    
    return pd.read_csv(path, sep=sep, decimal=decimal)


COLUMN_RENAME_MAP = {
    "Time": "Time",
    "Flow rate (Arith. Mean)": "Flow rate (Mean)",
    "Pressure before pump (Arith. Mean)": "TS inlet pressure (Mean)",
    "Pressure before pump (Arith. Mean)": "TS outlet pressure (Mean)",
    "Pressure after pump (Arith. Mean)": "Pump outlet pressure (Mean)",
    "Temperature TS inlet (Arith. Mean)": "Temperature TS inlet (Mean)",
    "Temperature TS outlet (Arith. Mean)": "Temperature TS outlet (Mean)",
    "Tank temperature (Arith. Mean)": "Tank temperature (Mean)",
    "Bypass temperature (Arith. Mean)": "Bypass temperature (Mean)",
    "Differential pressure (Arith. Mean)": "Differential pressure (Mean)",
}

def standardize_column_names(df):
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df

def load_raw_data(path="../data/raw_data/data1/*.csv"):
    '''Load and concatenate CSV files from a given path'''

    files = sorted(glob.glob(path))
    
    df_list = [read_unify_data(f) for f in files]

    df = pd.concat(df_list)

    df['Time'] = df['Time'].str.split(' ').str[1] if ' ' in df['Time'].iloc[0] else df['Time']

    df['Time'] = df['Time'].str.replace(',', '.', regex=False)

    df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S.%f")

    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    df = standardize_column_names(df)

    # Drop any columns that are unnamed
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df

def load_data(path="../data/raw_data/data1/*.csv"):
    '''Load and concatenate CSV files from a given path'''

    files = sorted(glob.glob(path))
    
    df_list = [read_unify_data(f) for f in files]

    df = pd.concat(df_list)

    df['Time'] = df['Time'].str.split(' ').str[1] if ' ' in df['Time'].iloc[0] else df['Time']

    df['Time'] = df['Time'].str.replace(',', '.', regex=False)

    df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S.%f")
    
    # Sort by Time 
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    df.reset_index(drop=True, inplace=True)

    df = standardize_column_names(df)

    # Drop any columns that are unnamed
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def save_data(data: dict, dataset_name: str, base_path="../data/processed_data/"):
    '''Save datasets to CSV files'''
    for key, df in data.items():
        df.to_csv(f"{base_path}{dataset_name}_{key}.csv", index=False)
    print(f"💾 Saved data with base name: {dataset_name}")
