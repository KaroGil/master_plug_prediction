"""
This module provides windowing functions for time series data.
"""


def create_sliding_windows(data, window_size, step_size):
    """
    Create sliding windows from time series data.

    Parameters:
    data (pd.DataFrame): Input time series data.
    window_size (int): Size of each window.
    step_size (int): Step size between windows.

    Returns:
    list of pd.DataFrame: List containing the sliding windows.
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        windows.append(data.iloc[start:end].reset_index(drop=True))
    return windows

