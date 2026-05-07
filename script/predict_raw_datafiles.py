"""
Script to run predictions on raw data files using the best trained model. 
The script loads the raw data, preprocesses it, makes predictions, and saves the results to a CSV file. 
It also visualizes the predicted values for analysis.
"""

import joblib
import argparse
import pandas as pd
from pathlib import Path
from script.helper_methods.data_loader import load_raw_data
from script.helper_methods.data_preprocessing import preprocess_data_predict
from script.helper_methods.data_visualization.predictions import visualize_predicted_vs_true


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=int, help="Dataset number, e.g. 1 for data1")
    return parser.parse_args()


def run_predictions(
    data_path="./data/raw_data/data1/*.csv",
    model_path="./models/best_model.joblib",
    output_path="./predictions/predictions.csv",
):
    # Load
    df = load_raw_data(data_path)

    log_id = df["LogId"][0]

    # Preprocess (no labels)
    X,_ = preprocess_data_predict(df, dataset_name="Loaded Data", add_label=False)

    # Predict
    model = joblib.load(model_path)
    predictions = model.predict(X)

    # Save
    results = pd.DataFrame({
        "prediction": predictions
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{output_path}_{log_id}.csv", index=False)
    print(f"Predictions saved to {output_path}_{log_id}.csv")


    visualize_predicted_vs_true(X, predictions, plotLabel=False)

if __name__ == "__main__":
    args = parse_args()
    data_path = f"./data/raw_data/data{args.dataset}/*.csv"
    run_predictions(data_path=data_path)