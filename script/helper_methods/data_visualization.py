import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


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


def visualize_flow_rate(data, name=None):
    '''Visualize flow rate and pump outlet pressure over time'''

    plt.figure(figsize=(12,6))
    plt.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data["Flow rate (Mean)"], label="Flow rate")
    plt.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data["Pump outlet pressure (Mean)"], label="Pump outlet pressure")

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


def visualize_plug_event(data, plug_column="Plug", anomalies=False, name=None):
    '''Visualize Plug=1 events on flow rate and pump outlet pressure'''

    plt.figure(figsize=(12,6))

    # Plot all data
    plt.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data["Flow rate (Mean)"], label="Flow rate", alpha=0.5)
    plt.plot(data["Elapsed_seconds"] if "Elapsed_seconds" in data.columns else data.index, data["Pump outlet pressure (Mean)"], label="Pump outlet pressure", alpha=0.5)

    # Highlight Plug=1 events
    plug_events = data[data[plug_column] == 1]
    plt.scatter(plug_events.index, plug_events["Flow rate (Mean)"], color="red", label=f"{plug_column}=1 (Flow)", zorder=5)
    plt.scatter(plug_events.index, plug_events["Pump outlet pressure (Mean)"], color="orange", label=f"{plug_column}=1 (Pressure)", zorder=5)

    # Highlight anomalies
    if anomalies and "Anomaly" in data.columns:
        anomaly_events = data[data["Anomaly"] == 1]
        plt.scatter(anomaly_events.index, anomaly_events["Flow rate (Mean)"], color="purple", label="Anomaly (Flow)", zorder=6, marker='x')
        plt.scatter(anomaly_events.index, anomaly_events["Pump outlet pressure (Mean)"], color="brown", label="Anomaly (Pressure)", zorder=6, marker='x')

    plt.xlabel("Elapsed_seconds")
    plt.ylabel("Value")
    plt.title(f"Plug_future=1 Events for {name}" if name else "Plug_future=1 Events")

    if name == "Labled Dataset":
        import matplotlib.dates as mdates
        plt.xlabel("Timestamp")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
    plt.legend()
    plt.show()


def visualize_predicted_vs_true(df, y_pred, anomalies=False, model_name=None, plotLabel=True):
    plt.figure(figsize=(12,6))

    print("DEBUG: df columns:", df.columns)

    flow_col = "Flow rate (Mean)"
    pressure_col = "Pump outlet pressure (Mean)"

    if flow_col not in df.columns:
        flow_col = "Flow rate (Mean)_mean"

    if pressure_col not in df.columns:
        pressure_col = "Pump outlet pressure (Mean)_mean"

    # Plot all data
    plt.plot(df.index, df[flow_col], label="Flow rate", alpha=0.5)
    plt.plot(df.index, df[pressure_col], label="Pump outlet pressure", alpha=0.5) if pressure_col in df.columns else None

    # Highlight true Plug_future=1 events
    if plotLabel:
        plug_events = df[df["Plug_future"] == 1] if "Plug_future" in df.columns else df[df["Plug_future_std"] == 1]
        plt.scatter(plug_events.index, plug_events[flow_col], color="red", label="Plug=1 (Flow)", zorder=5) 
        plt.scatter(plug_events.index, plug_events[pressure_col], color="orange", label="Plug=1 (Pressure)", zorder=5) if pressure_col in df.columns else None
        
    # Highlight anomalies
    if anomalies and ("Anomaly" in df.columns or "Anomaly_mean" in df.columns):
        anomaly_events = df[df["Anomaly"] == 1] if "Anomaly" in df.columns else df[df["Anomaly_mean"] == 1]
        plt.scatter(anomaly_events.index, anomaly_events[flow_col], color="purple", label="Anomaly (Flow)", zorder=6, marker='x') 
        plt.scatter(anomaly_events.index, anomaly_events[pressure_col], color="brown", label="Anomaly (Pressure)", zorder=6, marker='x') if pressure_col in df.columns else None 

    # Highlight predicted Plug events
    plug_events = df[y_pred == 1]
    plt.scatter(plug_events.index, plug_events[flow_col], color="yellow", label="Predicted plug (Flow)", zorder=7, marker='.') 
    plt.scatter(plug_events.index, plug_events[pressure_col], color="green", label="Predicted plug (Pressure)", zorder=7, marker='.')  if pressure_col in df.columns else None


    plt.xlabel("Elapsed_seconds")
    plt.ylabel("Value")
    model_name_str = f" by {model_name}" if model_name else ""
    plt.title(f"Predicted vs True Plug=1 Events{model_name_str}")
    plt.legend()
    plt.show()


def plot_feature_importance(feat_imp):
    '''Plot feature importance from a trained model'''

    feat_imp_sorted = feat_imp.sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feat_imp_sorted.values, y=feat_imp_sorted.index, palette="viridis", hue=feat_imp_sorted.values, legend=False)
    plt.title('Feature Importance from RandomForest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
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


def plot_anomaly_score_distribution(model, data):
    '''Plot distribution of anomaly scores'''

    scores = -model.score_samples(data)
    plt.hist(scores, bins=50)
    plt.title(f"{type(model).__name__} anomaly score distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.show()


def feature_importnace_anomaly(model):
    '''Plot feature importance for model'''

    importances = np.array([tree.feature_importances_ for tree in model.estimators_])

    importances = importances.mean(axis=0)

    plt.bar(range(len(importances)), importances)
    plt.title(f"{type(model).__name__} Feature Importance")
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
    plt.show()


def plot_anomaly_distribution(preds, model_name):
    unique, counts = np.unique(preds, return_counts=True)
    plt.bar(unique, counts)
    plt.xticks(unique, ['Normal', 'Anomaly'])
    plt.title(f"Anomaly Distribution for {model_name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()