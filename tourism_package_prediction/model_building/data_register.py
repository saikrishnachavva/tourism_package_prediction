# =============================================
# Data Registration & Validation Script
# Tourism Package Prediction - MLOps Pipeline
# =============================================
# This script validates the dataset CSV file stored
# in the repository's data/ folder. It checks that
# all expected columns are present and prints a
# comprehensive summary of the dataset.

import pandas as pd
import os

# ----- Load Dataset from Repository Data Folder -----
data_path = os.path.join("tourism_package_prediction", "data", "tourism.csv")
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully from: {data_path}")

# ----- Data Validation: Check Expected Columns -----
expected_columns = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

# Check if all expected columns are present
missing_cols = [col for col in expected_columns if col not in df.columns]
extra_cols = [col for col in df.columns if col not in expected_columns]

if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")
else:
    print("All expected columns are present in the dataset.")

if extra_cols:
    print(f"Extra columns found (not in expected list): {extra_cols}")

# ----- Print Dataset Summary -----
print(f"\n{'='*60}")
print(f"DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n  Columns: {list(df.columns)}")
print(f"\n  Data Types:\n{df.dtypes.to_string()}")
print(f"\n  Missing Values:\n{df.isnull().sum().to_string()}")
print(f"\n  Total Missing Values: {df.isnull().sum().sum()}")
print(f"\n  Target Distribution (ProdTaken):")
print(f"{df['ProdTaken'].value_counts().to_string()}")
print(f"\n  Target Percentage:")
print(f"{(df['ProdTaken'].value_counts(normalize=True) * 100).round(2).to_string()}")
print(f"{'='*60}")
print("Data registration and validation complete.")
