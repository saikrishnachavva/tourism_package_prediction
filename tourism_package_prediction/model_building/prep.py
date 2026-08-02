# =============================================
# Data Preparation Script
# Tourism Package Prediction - MLOps Pipeline
# =============================================
# This script loads the dataset from the repository's
# data folder, performs data cleaning, label encoding,
# and splits the data into training and testing sets.
# The splits are saved as CSV files locally; the pipeline
# uploads them as GitHub Actions workflow artifacts.

# Import necessary libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================
# LOAD DATASET FROM REPOSITORY DATA FOLDER
# =============================================

data_path = os.path.join("tourism_package_prediction", "data", "tourism.csv")
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully from: {data_path}")
print(f"   Shape: {df.shape}")

# =============================================
# DATA CLEANING
# =============================================

# 1. Drop the unnamed index column (first column) and CustomerID
# The unnamed column is a redundant row index from the original CSV
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    print("Dropped 'Unnamed: 0' column (redundant index).")

# CustomerID is a unique identifier - not useful for modeling
df.drop(columns=['CustomerID'], inplace=True)
print("Dropped 'CustomerID' column (unique identifier, not predictive).")

# 2. Standardize Gender values ('Fe Male' -> 'Female')
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
print("Standardized Gender values: 'Fe Male' → 'Female'")

# 3. Standardize MaritalStatus ('Unmarried' -> 'Single')
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
print("Standardized MaritalStatus values: 'Unmarried' → 'Single'")

# 4. Handle Missing Values
print(f"\n Missing values before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# Numerical columns - fill with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   Filled '{col}' missing values with median: {median_val}")

# Categorical columns - fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"   Filled '{col}' missing values with mode: '{mode_val}'")

print(f"\n Total missing values after cleaning: {df.isnull().sum().sum()}")

# =============================================
# LABEL ENCODING
# =============================================

# Encode categorical columns using LabelEncoder
label_encoder = LabelEncoder()
cat_columns_to_encode = ['TypeofContact', 'Occupation', 'Gender',
                         'ProductPitched', 'MaritalStatus', 'Designation']

print("\n Label Encoding Categorical Columns:")
for col in cat_columns_to_encode:
    original_values = sorted(df[col].unique())
    df[col] = label_encoder.fit_transform(df[col])
    encoded_values = sorted(df[col].unique())
    print(f"   {col}: {original_values} → {encoded_values}")

# =============================================
# TRAIN-TEST SPLIT
# =============================================

# Define target variable
target_col = 'ProdTaken'

# Split into features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform stratified train-test split (80/20)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Train-Test Split:")
print(f"   Train set: {Xtrain.shape[0]} samples ({Xtrain.shape[0]/len(df)*100:.1f}%)")
print(f"   Test set:  {Xtest.shape[0]} samples ({Xtest.shape[0]/len(df)*100:.1f}%)")
print(f"\n   Train target distribution:\n{ytrain.value_counts().to_string()}")
print(f"\n   Test target distribution:\n{ytest.value_counts().to_string()}")
print(f"\n   Feature columns ({X.shape[1]}): {list(X.columns)}")

# =============================================
# SAVE SPLITS AS CSV (locally)
# =============================================
# These files will be uploaded as GitHub Actions
# workflow artifacts by the pipeline YAML.

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("\n Train/test splits saved locally as CSV files.")
print("   Files: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
