import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from script.helper_methods.config import get_config

# Load config
cfg = get_config()

seed = cfg["experiment"]["random_state"]
dataset_nr = cfg['data']['datasets']
target_col = cfg["data"]["target"]
test_sets = cfg["data"]["test_sets"]


def plot_feature_histograms(data, name=None):
    '''Histograms for each numeric feature'''

    data.select_dtypes(include=[np.number]).hist(bins=30, figsize=(15, 10), layout=(2, -1))
    plt.suptitle(f'Histograms of Features for {name}' if name else 'Histograms of Features', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_feature_boxplots(data, name=None):
    '''Boxplots for each numeric feature'''

    df_numeric = data.select_dtypes(include=[np.number])
    num_features = df_numeric.shape[1]
    cols = 2
    rows = (num_features + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(df_numeric.columns):
        sns.boxplot(x=df_numeric[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f'Boxplot of {col} for {name}' if name else f'Boxplot of {col}')
        axes[i].set_xlabel("")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualize_key_column(data, key="Flow rate (Mean)", name=None):
    '''Visualize flow rate and pump outlet pressure over time'''

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


def plot_flow_pressure_drop_temp(data, start_time=None, end_time=None, name=None):
    '''Plot Flow rate, Pump outlet pressure, Drop pressure and Temperature over time'''

    if start_time and end_time:
        data = data.loc[start_time:end_time]

    fig, ax = plt.subplots(figsize=(16,8))
    colors = plt.cm.tab20.colors 
    for i, col in enumerate(data.columns):
        ax.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data[col], label=col, color=colors[i % len(colors)])
    ax.set_xlabel("Elapsed_seconds")
    ax.set_ylabel("Value")
    ax.set_title(f"All Signals Over Time for {name}" if name else "All Signals Over Time")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    fig.tight_layout()
    plt.show()

    return fig


def visualize_plug_event(data, plug_column=target_col, key="Flow rate (Mean)", name=None, extra_id=0, ax=None):
    """
    Visualize a key signal (flow rate or pressure) with plug events highlighted, and optionally extra vertical lines for specific times based on extra_id.
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
        
        #14:22 – 3.9% 14:54 – 4.4%; 15:54 – 4.3% 17:55 – 3.0% 19:58 – 3.0% 20:38 – 3.0% TODO: remove this or fix frequencing in datast 18
        # ax.axvline(pd.to_datetime("2022-11-03 14:22"), color="purple", linestyle="--", label="Time = 14:22")
        # ax.axvline(pd.to_datetime("2022-11-03 14:54"), color="green", linestyle="--", label="Time = 14:54")
        # ax.axvline(pd.to_datetime("2022-11-03 15:54"), color="pink", linestyle="--", label="Time = 15:54")
        # ax.axvline(pd.to_datetime("2022-11-03 17:55"), color="yellow", linestyle="--", label="Time = 17:55")
        # ax.axvline(pd.to_datetime("2022-11-03 19:58"), color="gray", linestyle="--", label="Time = 19:58")
        # ax.axvline(pd.to_datetime("2022-11-03 20:38"), color="gray", linestyle="--", label="Time = 20:38")

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



def visualize_predicted_vs_true(df, y_pred, model_name=None, plotLabel=True, flow_rate_missing=False):
    plt.figure(figsize=(12,6))

    flow_col = "Flow rate (Mean)"
    pressure_col = "Pump outlet pressure (Mean)"

    if flow_col not in df.columns:
        if "flow_rate" in df.columns:
            flow_col = "flow_rate"
        else:
            flow_col = "Flow rate (Mean)_mean"

    if pressure_col not in df.columns:
        pressure_col = "Pump outlet pressure (Mean)_mean"

        if pressure_col not in df.columns:
            pressure_candidates = df.columns[df.columns.str.contains("press", case=False, na=False)]
            if len(pressure_candidates) > 0:
                pressure_col = pressure_candidates[0]
            else:
                pressure_col = None

    # Plot all data
    if not flow_rate_missing:
        plt.plot(df.index, df[flow_col], label="Flow rate", alpha=0.5)

    plt.plot(df.index, df[pressure_col], label="Pump outlet pressure", alpha=0.5) 

    # Highlight true target events
    if plotLabel:
        plug_events = df[df[target_col] == 1]
        if not flow_rate_missing:
            plt.scatter(plug_events.index, plug_events[flow_col], color="red", label="Plug=1 (Flow)", zorder=5)

        plt.scatter(plug_events.index, plug_events[pressure_col], color="orange", label="Plug=1 (Pressure)", zorder=5)
        
    # Highlight predicted Plug events
    plug_events = df[y_pred == 1]
    if not flow_rate_missing:
        plt.scatter(plug_events.index, plug_events[flow_col], color="yellow", label="Predicted plug (Flow)", zorder=7, marker='.') 
    
    plt.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.') 

    plt.xlabel("Elapsed_seconds")

    labels = []

    if flow_col and not flow_rate_missing:
        labels.append("Flow rate [kg/h]")
    if pressure_col:
        labels.append("Pressure [Pa]")

    ylabel = " / ".join(labels) if labels else "Sensor value"
    plt.ylabel(ylabel)

    model_name_str = f" by {model_name}" if model_name else ""
    plt.title(f"Predicted vs True Plug=1 Events{model_name_str}")
    plt.legend()
    plt.show()


def plot_correlation_matrix(X_train):
    '''Plot correlation matrix of features to identify relationships'''

    corr = X_train.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, standalone=True, ax=None, labels=None):
    '''Plot confusion matrix'''
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        _, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    if standalone:
        plt.show()


def get_y_scores(model, X):
    """Returns continuous scores for PR / Calibration curves."""
    
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        raise ValueError(
            f"Model {type(model).__name__} does not provide probability or decision scores."
        )


def plot_precision_recall_curve(y_true, y_scores, standalone=True, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()

    if standalone:
        plt.show()


def learning_curve(train_sizes, train_scores, val_scores, standalone=True, ax=None):
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(train_sizes, train_mean, 'o-', label="Training")
    ax.plot(train_sizes, val_mean, 'o-', label="Validation")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid()

    if standalone:
        plt.show()


def plot_calibration_curve(y_true, y_scores, n_bins=10, standalone=True, ax=None):
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Fraction")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid()

    if standalone:
        plt.show()


def plot_all(model, X_test, y_test, train_sizes, train_scores, val_scores):
    y_pred = model.predict(X_test)
    y_scores = get_y_scores(model, X_test)

    _, ax = plt.subplots(2, 2, figsize=(18, 10))

    plot_confusion_matrix(y_test, y_pred, standalone=False, ax=ax[0, 0])
    learning_curve(train_sizes, train_scores, val_scores, standalone=False, ax=ax[1, 0])
    plot_precision_recall_curve(y_test, y_scores, standalone=False, ax=ax[0, 1])
    plot_calibration_curve(y_test, y_scores, standalone=False, ax=ax[1, 1])

    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model,
    X: pd.DataFrame,
    y,
    horizon, 
    method: str = "auto",
    top_n: int = 25,
    scoring: str = "f1",
    n_repeats: int = 5,
    random_state: int = seed,
    drop_cols: tuple[str, ...] = ("LogId",),
    figsize=(9, 7),
):

    Xp = X.drop(columns=[c for c in drop_cols if c in X.columns]).copy()
    feature_names = Xp.columns.to_list()

    if method not in {"auto", "permutation", "model"}:
        raise ValueError("method must be one of: auto, permutation, model")

    use_model = False
    if method in {"auto", "model"}:
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
            use_model = True
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importances = np.abs(coef).ravel()
            use_model = True

    if method == "permutation" or (method == "auto" and not use_model):
        pi = permutation_importance(
            model, Xp, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        importances = pi.importances_mean
        title = f"Permutation importance (scoring={scoring})"
    else:
        title = "Model-based feature importance"

    # Build ranking table
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    #plt.show() #TODO Test this

    plt.savefig(f"/plots/feature_importance_{horizon}.png", dpi=300)
    plt.close()

    return imp_df.reset_index(drop=True)


### VISUALIZING RESULTS FOR PREDICT_ALL ###
def plot_one(df, y_pred, figureNum, y, nrows, ncols, dataset_id = None, flow_col="Flow rate (Mean)", pressure_col="Pump outlet pressure (Mean)", show_flow=True):
    if flow_col not in df.columns:
        flow_col = "flow_rate"

    if not show_flow:
        flow_col = None
    
    if pressure_col not in df.columns:
        pressure_candidates = df.columns[df.columns.str.contains("press", case=False, na=False)]
    
        if len(pressure_candidates) == 0:
            pressure_col = None
        elif "TS outlet pressure (Mean)_std" in pressure_candidates:
            pressure_col = "TS outlet pressure (Mean)_std"
        else:
            pressure_col = pressure_candidates[0]

    # Determine if this dataset is part of training or test set for labeling the plot
    shown_id = dataset_id if dataset_id is not None else figureNum
    dataset_nrs = list(dataset_nr)
    test_set_nr = max(dataset_nrs)
    test_set_nrs = test_sets if test_sets else [test_set_nr]
    train_set_nrs = [nr for nr in dataset_nrs if nr not in test_set_nrs]

    # Create subplot
    plt.subplot(nrows, ncols, figureNum)
    
    if flow_col in df.columns:
        plt.plot(df.index, df[flow_col], label="Flow rate", alpha=0.5)
    elif pressure_col in df.columns: 
        plt.plot(df.index, df[pressure_col], label="Pump outlet pressure", alpha=0.5)

    # Highlight true Plug events, on training data
    true_plug_events = df[y == 1]
    if shown_id not in test_set_nrs:
        if flow_col in df.columns: 
            plt.scatter(true_plug_events.index, true_plug_events[flow_col], color="red", label="True Plug=1 (Flow)", zorder=6, marker='x')
        if pressure_col in df.columns:
            plt.scatter(true_plug_events.index, true_plug_events[pressure_col], color="blue", label="True Plug=1 (Pressure)", zorder=6, marker='x')
    
    # Highlight predicted Plug events, on all data
    plug_events = df[y_pred == 1]
    if flow_col in df.columns:
        plt.scatter(plug_events.index, plug_events[flow_col], color="yellow", label="Predicted plug (Flow)", zorder=7, marker='.')
    if pressure_col in df.columns:
        plt.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.')  
     
    plt.xlabel("Elapsed_seconds")

    labels = []

    if flow_col and show_flow:
        labels.append("Flow rate [kg/h]")
    if pressure_col:
        labels.append("Pressure [Pa]")

    ylabel = " / ".join(labels) if labels else "Sensor value"
    plt.ylabel(ylabel)

    if shown_id in test_set_nrs:
        plt.title(f"Data nr {shown_id} [test set]")
    else:
        plt.title(f"Data nr {shown_id}" if shown_id not in train_set_nrs else f"Data nr {shown_id} [training set]")
    plt.legend()


## Horizon

def plot_test_f1_vs_horizon(
    horizons,
    test_scores,
    invert_xaxis=False,
    figsize=(8,5)
):
    horizons = np.array(horizons)
    test_scores = np.array(test_scores)

    plt.figure(figsize=figsize)

    plt.plot(horizons, test_scores,
             marker="o", linewidth=2, label="Random Forest")

    plt.xlabel("Prediction Horizon (s before plug)")
    plt.ylabel("F1 Score (Test Set)")
    plt.title("Test F1 vs Prediction Horizon")
    plt.legend()
    plt.grid(True)

    if invert_xaxis:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    #plt.show() TODO Test this

    plt.savefig("/plots/horizon_test/f1_scores_line_plot.png", dpi=300)
    plt.close()


def plot_test_f1_vs_horizon_bar(
    horizons,
    test_scores,
    invert_xaxis=False,
    figsize=(8,5)
):
    horizons = np.array(horizons)
    test_scores = np.array(test_scores)

    plt.figure(figsize=figsize)

    x = np.arange(len(horizons)) 

    plt.bar(x, test_scores, width=0.6, alpha=0.8)

    best_score = np.max(test_scores)
    plt.axhline(best_score, color="red", linestyle="--", linewidth=2)
    plt.text(len(x)-0.5, best_score + 0.005,
             f"Best: {best_score:.3f}",
             color="red", ha="right")
    
    for i in range(len(x)):
        plt.text(i, test_scores[i] // 2, f"{test_scores[i]:.3f}", ha='center')

    plt.xlabel("Prediction Horizon (s before plug)")
    plt.ylabel("F1 Score (Test Set)")
    plt.title("Test F1 vs Prediction Horizon")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(x, horizons)

    if invert_xaxis:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    #plt.show() TODO Test this
    plt.savefig("/plots/horizon_test/f1_scores_bar_plot.png", dpi=300)
    plt.close()
