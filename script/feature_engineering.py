def rolling_features(df, window=10, functions=['mean', 'std', 'min', 'max']):
    '''
    Generate rolling features for the given DataFrame.
    '''
    # Create a copy of the DataFrame to avoid modifying the original data
    df_rolling = df.copy()

    # Exclude non-numeric columns
    df_rolling = df_rolling.select_dtypes(include=['number'])

    # Generate rolling features
    for col in df_rolling.columns:
        rolling = df_rolling[col].rolling(window=window, min_periods=1)

        for func in functions:
            col_name = f"{col}_rolling_{func}_{window}"
            df_rolling[col_name] = getattr(rolling, func)().bfill()

    return df_rolling
