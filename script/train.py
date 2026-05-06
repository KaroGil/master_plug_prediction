"""
This script is responsible for training the models on the labeled datasets. 
It loads the datasets, preprocesses the data, and trains the models. 
The script also includes a step to delete any preprocessed datasets in the predict folder to avoid confusion with new datasets. 
The preprocessing includes creating future targets, feature engineering, feature reduction, and splitting the data into training and test sets.
The models are evaluated using the test set, and the results are printed out.
The trained models are then saved for later use in prediction and evaluation.
"""
from script.data_prep import load_and_preprocess_data
from script.helper_methods.data_modeling import model_data

# Preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train and evaluate models
model_data(X_train, y_train, X_test, y_test)
