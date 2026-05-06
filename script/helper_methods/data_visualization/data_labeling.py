"""
Helper methods used for visualizing used during the labeling process.
"""

import pandas as pd  
import matplotlib.pyplot as plt
from script.helper_methods.config import get_config

# Config for plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Load config
cfg = get_config()

target_col = cfg["data"]["target"]

def visualize_key_column(data, key="Flow rate (Mean)", name=None):
    """
    Visualize flow rate and pump outlet pressure over time
    Used when labeling data. 
    """

    if key not in data.columns:
        raise ValueError(f"Column '{key}' not found in dataset.")

    plt.figure(figsize=(12,6))
    plt.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data[key], label=key)

    plt.xlabel("Elapsed_seconds")
    plt.ylabel("Value")
    plt.title(f"Flow rate & Pump outlet pressure over time for {name}" if name else "Flow rate & Pump outlet pressure over time")
    if name == "Labled Dataset":
        import matplotlib.dates as mdates
        plt.xlabel("Timestamp")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.legend()
    plt.show()


def visualize_plug_event(data, plug_column=target_col, key="Flow rate (Mean)", name=None, extra_id=0, ax=None):
    """
    Visualize a key signal (flow rate or pressure) with plug events highlighted, and optionally extra vertical lines for specific times based on extra_id.
    Used when labeling data.
    - `data`: DataFrame containing the data to plot
    - `plug_column`: Column name indicating plug events (default is target_col)
    - `key`: Column name of the signal to plot (default is "Flow rate (Mean)")
    - `name`: Optional name for the plot title
    - `extra_id`: If set to 0, nothing extra is plotted. Other numbers might correspond to specific extra lines to plot (e.g. for dataset 7)
    - `ax`: Optional matplotlib axis to plot on. If None, a new figure and axis will be created   
    """
    possible_keys = [key, f"{key}_mean"]
    actual_key = next((col for col in possible_keys if col in data.columns), None)

    if actual_key is None:
        raise ValueError(f"None of these columns were found: {possible_keys}")

    created_figure = False
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
        created_figure = True

    x = data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index

    # Plot signal
    ax.plot(x, data[actual_key], label=actual_key, alpha=0.5)

    # Plug events
    plug_events = data[data[plug_column] == 1]

    if not plug_events.empty:
        x_plug = plug_events["Elapsed_seconds"] if "Elapsed_seconds" in plug_events.columns else plug_events.index

        ax.scatter(
            x_plug,
            plug_events[actual_key],
            color="red",
            label=f"{plug_column}=1",
            zorder=5,
            s=15
        )

    if extra_id == 3:
        # Indicate end of startup noise, first stabilize at 400kg/h
        ax.axvline(pd.to_datetime("2021-09-01 10:59:21.171000"), color="black", linestyle="--", label="Time = 15:37:26")

        #Spike
        ax.axvline(pd.to_datetime("2021-09-01 11:28:25.069000"), color="black", linestyle="--", label="Time = 15:37:26")
        ax.axvline(pd.to_datetime("2021-09-01 11:36:49.038000"), color="black", linestyle="--", label="Time = 15:37:26")

        # End of experiment spike
        ax.axvline(pd.to_datetime("2021-09-01 12:41:33.813000"), color="black", linestyle="--", label="Time = 15:37:26")

        #1 – 3.9% (-003 min), 2 – 5.7% (006 min), 3 – 4% (012 min), 4 – 3.8% (018 min), 5 – 2.5% (025 min), 6 – (033 min)
        base = data.index[0]

        ax.axvline(base + pd.Timedelta(minutes=-3), linestyle="--", color="red", label="-003 min")
        ax.axvline(base + pd.Timedelta(minutes=6), linestyle="--", color="orange", label="006 min")
        ax.axvline(base + pd.Timedelta(minutes=12), linestyle="--", color="yellow", label="012 min")
        ax.axvline(base + pd.Timedelta(minutes=18), linestyle="--", color="green", label="018 min")
        ax.axvline(base + pd.Timedelta(minutes=25), linestyle="--", color="blue", label="025 min")
        ax.axvline(base + pd.Timedelta(minutes=33), linestyle="--", color="purple", label="033 min")

    if extra_id == 4:
        ax.axvline(pd.to_datetime("1900-01-01 15:37:26"), color="purple", linestyle="--", label="Time = 15:37:26")
        ax.axvline(pd.to_datetime("1900-01-01 16:17:41"), color="green", linestyle="--", label="Time = 16:17:41")
        ax.axvline(pd.to_datetime("1900-01-01 16:20:34"), color="pink", linestyle="--", label="Time = 16:20:34")
        ax.axvline(pd.to_datetime("1900-01-01 16:43:20"), color="yellow", linestyle="--", label="Time = 16:43:20")

    if extra_id == 7:
        ax.axvline(pd.to_datetime("1900-01-01 11:22"), color="purple", linestyle="--", label="Time = 11:22")
        ax.axvline(pd.to_datetime("1900-01-01 11:52"), color="green", linestyle="--", label="Time = 11:52")
        ax.axvline(pd.to_datetime("1900-01-01 12:52"), color="pink", linestyle="--", label="Time = 12:52")
        ax.axvline(pd.to_datetime("1900-01-01 14:58"), color="yellow", linestyle="--", label="Time = 14:58")
        ax.axvline(pd.to_datetime("1900-01-01 17:00"), color="gray", linestyle="--", label="Time = 17:00")

    if extra_id == 14:
        # Video starting times
        ax.axvline(pd.to_datetime("1900-01-01 11:28"), color="black", linestyle="--", label="Time = 11:28, video start")
        ax.axvline(pd.to_datetime("1900-01-01 16:26"), color="black", linestyle="--", label="Time = 16:26, video start")
        
        # Reaches target flowrate 1400 kg/h
        ax.axvline(pd.to_datetime("1900-01-01 11:46:21.233000"), color="yellow", linestyle="--", label="Time = 11:46, reaches target flowrate 1400 kg/h")

        # Plummet point, flowrate drops to 0
        ax.axvline(pd.to_datetime("1900-01-01 17:01:44.233000"), color="red", linestyle="--", label="Time = 17:01")

    if extra_id == 17:
        ax.axvline(pd.to_datetime("2021-12-07 10:43:03.843000"), color="yellow", linestyle="--", label="Time = 10:43, reaches target flowrate 1500 kg/h")
        ax.axvline(pd.to_datetime("2021-12-07 11:06:58.754000"), color="yellow", linestyle="--", label="Time = 11:06, reaches target flowrate 1500 kg/h")
        ax.axvline(pd.to_datetime("2021-12-07 12:22:37.470000"), color="red", linestyle="--", label="Time = 12:24, plummet point")

    if extra_id == 18:
        # Video starting times
        ax.axvline(pd.to_datetime("2022-11-03 14:06"), color="black", linestyle="--", label="Time = 14:06, video start")
        
        # Reaches target flowrate 400 kg/h
        ax.axvline(pd.to_datetime("2022-11-03 14:20:05"), color="yellow", linestyle="--", label="Time = 11:46, reaches target flowrate 400 kg/h")

        # End spike 
        ax.axvline(pd.to_datetime("2022-11-03 17:26:46.000"), color="red", linestyle="--", label="Time = 17:26, end spike")
    
    if extra_id == 24:
        # Video starts
        ax.axvline(pd.to_datetime("1900-01-01 13:43"), color="black", linestyle="--", label="Time = 13:43, video start")
        ax.axvline(pd.to_datetime("1900-01-01 15:17"), color="black", linestyle="--", label="Time = 15:17, video start")
        ax.axvline(pd.to_datetime("1900-01-01 16:28"), color="black", linestyle="--", label="Time = 16:28, video start")
        

        # Reached target flow rate 900 kg/h
        ax.axvline(pd.to_datetime("1900-01-01 13:57:42.474000"), color="yellow", linestyle="--", label="Time = 14:44, reaches target flowrate 900 kg/h")

        # Plummet
        ax.axvline(pd.to_datetime("1900-01-01 16:29:08.474"), color="red", linestyle="--", label="Time = 16:29:08, plummet start")
        ax.axvline(pd.to_datetime("1900-01-01 16:35:50.474"), color="red", linestyle="--", label="Time = 16:29:11, plummet end")
        
        # 13:56 – 13.4% 18:00 – 13.1%
        ax.axvline(pd.to_datetime("1900-01-01 13:56"), color="pink", linestyle="--", label="Time = 13:56")
        ax.axvline(pd.to_datetime("1900-01-01 18:00"), color="pink", linestyle="--", label="Time = 18:00")

    ax.set_title(f"{name}" if name else actual_key)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=6)

    if created_figure:
        plt.tight_layout()
        plt.show()

