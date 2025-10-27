"""
Feature engineering helpers for rolling-window / realtime usage.

Provides:
- build_rolling_features(df, cols, window, min_periods)
- make_window_features(window_df, cols)
- time_since_last_event(df, event_col)
- add_time_features(df, ts_col=None)
- fit_scaler / transform_with_scaler helpers (StandardScaler)
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _fft_stats(arr: np.ndarray, sample_rate: float = 1.0) -> Tuple[float, float, float]:
    """Return (spectral_centroid, peak_freq, spectral_energy) for 1D array."""
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n == 0 or np.all(np.isnan(arr)):
        return (np.nan, np.nan, np.nan)
    # remove mean to emphasize dynamics
    arr = arr - np.nanmean(arr)
    # zero-pad small windows to at least length 2 for rfft
    if n < 2:
        return (np.nan, np.nan, np.nan)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mags = np.abs(np.fft.rfft(arr, n=n))
    energy = np.sum(mags ** 2)
    if np.sum(mags) == 0:
        centroid = 0.0
    else:
        centroid = (freqs * mags).sum() / mags.sum()
    peak_idx = np.argmax(mags)
    peak_freq = freqs[peak_idx]
    return float(centroid), float(peak_freq), float(energy)


def make_window_features(window_df: pd.DataFrame,
                         cols: Optional[List[str]] = None,
                         sample_rate: float = 1.0) -> pd.DataFrame:
    """
    Compute aggregated features for a single window (DataFrame).
    Returns a single-row DataFrame with features derived from `cols`.
    If cols is None, uses all numeric cols in window_df.
    Features per column: mean, std, min, max, skew, kurt, delta (last-first),
    trend (delta/len), last_value, spectral_centroid, spectral_peak_freq, spectral_energy.
    """
    if cols is None:
        cols = window_df.select_dtypes(include=[np.number]).columns.tolist()
    feats = {}
    n = len(window_df)
    for c in cols:
        s = window_df[c].astype(float).values
        if n == 0:
            vals = np.array([np.nan])
        else:
            vals = s
        mean = np.nanmean(vals)
        std = np.nanstd(vals, ddof=1) if n > 1 else 0.0
        vmin = np.nanmin(vals) if n > 0 else np.nan
        vmax = np.nanmax(vals) if n > 0 else np.nan
        skew = pd.Series(vals).skew() if n > 2 else np.nan
        kurt = pd.Series(vals).kurtosis() if n > 3 else np.nan
        last = vals[-1] if n > 0 else np.nan
        first = vals[0] if n > 0 else np.nan
        delta = (last - first) if (n > 0 and not np.isnan(last) and not np.isnan(first)) else np.nan
        trend = delta / float(n) if n > 0 else np.nan
        centroid, peak_freq, energy = _fft_stats(vals, sample_rate=sample_rate)

        prefix = c.replace(" ", "_")
        feats[f"{prefix}_mean"] = mean
        feats[f"{prefix}_std"] = std
        feats[f"{prefix}_min"] = vmin
        feats[f"{prefix}_max"] = vmax
        feats[f"{prefix}_skew"] = skew
        feats[f"{prefix}_kurtosis"] = kurt
        feats[f"{prefix}_last"] = last
        feats[f"{prefix}_delta"] = delta
        feats[f"{prefix}_trend"] = trend
        feats[f"{prefix}_spec_centroid"] = centroid
        feats[f"{prefix}_spec_peak_freq"] = peak_freq
        feats[f"{prefix}_spec_energy"] = energy

    return pd.DataFrame([feats])


def build_rolling_features(df: pd.DataFrame,
                           cols: Optional[List[str]] = None,
                           window: int = 50,
                           min_periods: Optional[int] = None,
                           sample_rate: float = 1.0) -> pd.DataFrame:
    """
    Build rolling-window aggregate features for entire DataFrame.
    - df: time-ordered DataFrame (index should be time or monotonic)
    - cols: list of numeric columns to aggregate (defaults to all numeric)
    - window: rolling window size (in rows)
    - min_periods: minimum observations in window to compute features (defaults to window)
    Returns DataFrame aligned with original index; rows with insufficient data get NaNs.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if min_periods is None:
        min_periods = window

    features_list = []
    # iterate windows efficiently using rolling.apply isn't trivial for custom features -> iterate
    # for typical sizes this is acceptable; for large data optimize with numba/C extension if needed.
    for end_idx in range(len(df)):
        start_idx = end_idx - window + 1
        if start_idx < 0:
            cur_window = df.iloc[0:end_idx + 1]
        else:
            cur_window = df.iloc[start_idx:end_idx + 1]

        if len(cur_window) < min_periods:
            features_list.append(pd.Series({}))  # will create NaN row later
            continue

        feats = make_window_features(cur_window, cols=cols, sample_rate=sample_rate)
        features_list.append(feats.iloc[0])

    features_df = pd.DataFrame(features_list, index=df.index)
    # ensure consistent column ordering / types
    return features_df


def time_since_last_event(df: pd.DataFrame,
                          event_col: str,
                          new_col: Optional[str] = None) -> pd.DataFrame:
    """
    Compute number of rows since last event (event_col == 1). For rows with event, value is 0.
    Adds column to df (or returns Series if you prefer).
    """
    if new_col is None:
        new_col = f"{event_col}_since_last"
    s = pd.Series(0, index=df.index)
    ev = df[event_col].fillna(0).astype(bool)
    last = -np.inf
    count = []
    cnt = np.nan
    # vectorized approach: use forward-fill on mask of events
    idx = np.arange(len(ev))
    last_event_idx = np.where(ev)[0]
    if last_event_idx.size == 0:
        s[:] = np.nan
        df[new_col] = s
        return df
    # compute distances to last event
    last_seen = -1
    out = np.full(len(ev), np.nan, dtype=float)
    for i, flag in enumerate(ev):
        if flag:
            last_seen = i
            out[i] = 0.0
        else:
            if last_seen >= 0:
                out[i] = i - last_seen
            else:
                out[i] = np.nan
    df[new_col] = out
    return df


# def add_time_features(df: pd.DataFrame, ts_col: Optional[str] = None) -> pd.DataFrame:
#     """
#     Add simple cyclical/time features if timestamp available.
#     If ts_col is None, will try to use df.index if it's DatetimeIndex.
#     Adds: hour_sin, hour_cos, dayofweek_sin, dayofweek_cos
#     """
#     if ts_col is not None:
#         ts = pd.to_datetime(df[ts_col])
#     elif isinstance(df.index, pd.DatetimeIndex):
#         ts = df.index.to_series()
#     else:
#         raise ValueError("No timestamp column provided and index is not DatetimeIndex")

#     seconds_in_day = 24 * 60 * 60
#     seconds = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
#     hour_angle = 2 * np.pi * seconds / seconds_in_day
#     df["hour_sin"] = np.sin(hour_angle)
#     df["hour_cos"] = np.cos(hour_angle)

#     dow = ts.dt.dayofweek
#     dow_angle = 2 * np.pi * dow / 7
#     df["dow_sin"] = np.sin(dow_angle)
#     df["dow_cos"] = np.cos(dow_angle)
#     return df
