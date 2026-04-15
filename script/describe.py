from script.helper_methods.data_loader import load_dataset_artifact

# Load preprocessed data
data_path = "./data/processed_data/" 
dataset_name = load_dataset_artifact("LATEST", data_path) 
data = load_dataset_artifact(dataset_name["artifact_path"].strip(".joblib"), ".")
df = data["X_train"].copy()
feature_names = data["feature_names"]

# Display basic information
print("=" * 50)
print("DATA DESCRIPTION")
print("=" * 50)

print(f"\nShape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nUnique values in target variable:\n{data['y_train'].value_counts()}")
print(f"\nFeature names: {feature_names}")
print(f"\nDataset name: {dataset_name}")