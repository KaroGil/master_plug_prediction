"""
Script to select the best model for the dataset given preprocessed data is present. 
The selected model is saved for future use.
"""

from script.helper_methods.data_modeling import model_data
from script.helper_methods.data_loader import load_preprocessed_dataset

# Load data
X_train, y_train, X_test, y_test = load_preprocessed_dataset()

# Run the model selection process
model_data(X_train, y_train, X_test, y_test)