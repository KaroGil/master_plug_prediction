import pandas as pd
import os
PATH = os.path.join("models", "model_comparison_summary.csv")

df = pd.read_csv(PATH)
print("Model Comparison Summary:")
print("-------------------------")

for index, row in df.iterrows():
    print(f"{row['Model']:<30} {row['Best Validation F1 Score']:.4f}")

print("-------------------------", end="\n\n")
print(f"🏆 Best Model: {df.iloc[df['Best Validation F1 Score'].idxmax()]['Model']} with F1 Score {df['Best Validation F1 Score'].max():.4f}", end="\n\n")