import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


def plot_feature_histograms(data):
    '''Histograms for each numeric feature'''

    data.select_dtypes(include=[np.number]).hist(bins=30, figsize=(15, 10), layout=(2, -1))
    plt.suptitle('Histograms of Features', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_feature_boxplots(data):
    '''Boxplots for each numeric feature'''

    df_numeric = data.select_dtypes(include=[np.number])
    num_features = df_numeric.shape[1]
    cols = 2
    rows = (num_features + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(df_numeric.columns):
        sns.boxplot(x=df_numeric[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel("")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualize_flow_rate(data):
    '''Visualize flow rate and pump outlet pressure over time'''

    plt.figure(figsize=(12,6))
    plt.plot(data.index, data["Flow rate (Mean)"], label="Flow rate")
    plt.plot(data.index, data["Pump outlet pressure (Mean)"], label="Pump outlet pressure")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Flow rate & Pump outlet pressure over time")
    plt.legend()
    plt.show()


def plot_flow_pressure_drop_temp(data, start_time=None, end_time=None):
    '''Plot Flow rate, Pump outlet pressure, Drop pressure and Temperature over time'''

    if start_time and end_time:
        data = data.loc[start_time:end_time]

    plt.figure(figsize=(16,8))
    colors = plt.cm.tab20.colors  # Use a colormap for up to 20 columns, cycle if more
    for i, col in enumerate(data.columns):
        plt.plot(data.index, data[col], label=col, color=colors[i % len(colors)])
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("All Signals Over Time")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()


def visualize_plug_event(data, anomalies=False):
    '''Visualize Plug=1 events on flow rate and pump outlet pressure'''

    plt.figure(figsize=(12,6))

    # Plot all data
    plt.plot(data.index, data["Flow rate (Mean)"], label="Flow rate", alpha=0.5)
    plt.plot(data.index, data["Pump outlet pressure (Mean)"], label="Pump outlet pressure", alpha=0.5)

    # Highlight Plug=1 events
    plug_events = data[data["Plug"] == 1]
    plt.scatter(plug_events.index, plug_events["Flow rate (Mean)"], color="red", label="Plug=1 (Flow)", zorder=5)
    plt.scatter(plug_events.index, plug_events["Pump outlet pressure (Mean)"], color="orange", label="Plug=1 (Pressure)", zorder=5)

    # Highlight anomalies
    if anomalies and "Anomaly" in data.columns:
        anomaly_events = data[data["Anomaly"] == 1]
        plt.scatter(anomaly_events.index, anomaly_events["Flow rate (Mean)"], color="purple", label="Anomaly (Flow)", zorder=6, marker='x')
        plt.scatter(anomaly_events.index, anomaly_events["Pump outlet pressure (Mean)"], color="brown", label="Anomaly (Pressure)", zorder=6, marker='x')

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Plug=1 Events Highlighted on Flow and Pressure")
    plt.legend()
    plt.show()


def visualize_predicted_vs_true(df, y_pred, anomalies=False):
    plt.figure(figsize=(12,6))

    # Plot all data
    plt.plot(df.index, df["Flow rate (Mean)"], label="Flow rate", alpha=0.5)
    plt.plot(df.index, df["Pump outlet pressure (Mean)"], label="Pump outlet pressure", alpha=0.5)

    # Highlight Plug=1 events
    plug_events = df[df["Plug"] == 1]
    plt.scatter(plug_events.index, plug_events["Flow rate (Mean)"], color="red", label="Plug=1 (Flow)", zorder=5)
    plt.scatter(plug_events.index, plug_events["Pump outlet pressure (Mean)"], color="orange", label="Plug=1 (Pressure)", zorder=5)

    # Highlight anomalies
    if anomalies and "Anomaly" in df.columns:
        anomaly_events = df[df["Anomaly"] == 1]
        plt.scatter(anomaly_events.index, anomaly_events["Flow rate (Mean)"], color="purple", label="Anomaly (Flow)", zorder=6, marker='x')
        plt.scatter(anomaly_events.index, anomaly_events["Pump outlet pressure (Mean)"], color="brown", label="Anomaly (Pressure)", zorder=6, marker='x')

    # Highlight predicted Plug events
    plug_events = df[y_pred == 1]
    plt.scatter(plug_events.index, plug_events["Flow rate (Mean)"], color="yellow", label="Predicted plug (Flow)", zorder=5, marker='.')
    plt.scatter(plug_events.index, plug_events["Pump outlet pressure (Mean)"], color="green", label="Predicted plug (Pressure)", zorder=5, marker='.')


    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Plug=1 Events Highlighted on Flow and Pressure")
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