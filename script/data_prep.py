"""
Script to prepare the data for training and testing the models. 
It loads the datasets, preprocesses them, and splits them into training and testing sets. 
The preprocessed data is then saved and ready to be used for model training and evaluation.
Saved preprocessed data can be found in "data/preprocessed/".
"""
from script.helper_methods.data_preprocessing import preprocess_data
from script.helper_methods.data_loader import load_labeled_datasets
from script.helper_methods.config import get_config

cfg = get_config()
dataset_nr = cfg['data']['datasets']

def load_and_preprocess_data():
    """
    Load the labeled datasets, preprocess the data, and split it into training and testing sets.
    """
    
    # Load labeled datasets
    datasets = load_labeled_datasets()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(datasets, [f"data{i}" for i in dataset_nr])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()