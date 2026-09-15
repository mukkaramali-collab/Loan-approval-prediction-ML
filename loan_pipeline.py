"""
Loan Approval Prediction - Data Cleaning & Logistic Regression Model
Final Project Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ---------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------
df = pd.read_csv('loan_dataset.csv')
print("=" * 70)
print("STEP 1: RAW DATA LOADED")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Missing values before cleaning:\n{df.isnull().sum()}\n")

# ---------------------------------------------------------------
# 1b. CHECK & REMOVE DUPLICATE RECORDS
# ---------------------------------------------------------------
n_dupes = df.duplicated().sum()
n_dupes_by_id = df.duplicated(subset=['Loan_ID']).sum()
print("=" * 70)
print("STEP 1b: DUPLICATE RECORD CHECK")
print("=" * 70)
print(f"Fully duplicated rows: {n_dupes}")
print(f"Duplicate Loan_IDs: {n_dupes_by_id}")
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['Loan_ID'], keep='first')
print(f"Shape after duplicate removal: {df.shape}\n")

# ---------------------------------------------------------------
# 2. HANDLE MISSING VALUES
# ---------------------------------------------------------------
# Categorical columns -> fill with mode (most frequent value)
cat_cols_to_fill = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in cat_cols_to_fill:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

# Loan_Amount_Term -> fill with mode (it's a discrete/categorical-like numeric term, e.g. 360)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

# LoanAmount -> fill with median (robust to outliers/skew)
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

print("=" * 70)
print("STEP 2: MISSING VALUES HANDLED")
print("=" * 70)
print(f"Missing values after cleaning:\n{df.isnull().sum()}\n")
assert df.isnull().sum().sum() == 0, "There are still missing values!"

# ---------------------------------------------------------------
# 3. CLEAN / STANDARDIZE CATEGORICAL VALUES
# ---------------------------------------------------------------
# Dependents: '3+' -> 3, then convert to integer
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Credit_History: ensure it's an int (0/1) not a float
df['Credit_History'] = df['Credit_History'].astype(int)

# Loan_Amount_Term: this is really a discrete/categorical duration (months),
# not a continuous decimal -> store as int, not float
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype(int)

print("=" * 70)
print("STEP 3: INCORRECT DATA TYPES FIXED")
print("=" * 70)
print("Dependents: object -> int64 ('3+' mapped to 3)")
print("Credit_History: float64 -> int64")
print("Loan_Amount_Term: float64 -> int64\n")

# ---------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------
# Combine applicant and co-applicant income into one Total Income feature
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Income and loan amount are right-skewed -> log-transform to make them
# more suitable for a linear model like Logistic Regression
df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
df['TotalIncome_log'] = np.log1p(df['TotalIncome'])

# ---------------------------------------------------------------
# 5. ENCODE CATEGORICAL VARIABLES (Label / One-Hot)
# ---------------------------------------------------------------
# Binary Yes/No and Male/Female columns -> map to 0/1
binary_map = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
}
for col, mapping in binary_map.items():
    df[col] = df[col].map(mapping)

# Target variable: Loan_Status -> Y = 1 (Approved), N = 0 (Rejected)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Property_Area is nominal (no order) -> one-hot encode
df = pd.get_dummies(df, columns=['Property_Area'], prefix='Property_Area', drop_first=True)
# Convert the new one-hot columns from bool to int for a clean numeric dataset
onehot_cols = [c for c in df.columns if c.startswith('Property_Area_')]
df[onehot_cols] = df[onehot_cols].astype(int)

print("=" * 70)
print("STEP 3-5: CATEGORICAL CLEANING, FEATURE ENGINEERING & ENCODING DONE")
print("=" * 70)
print(f"Shape after encoding: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ---------------------------------------------------------------
# 6. DROP UNUSED COLUMNS
# ---------------------------------------------------------------
# Loan_ID is just an identifier, not a predictive feature
# We drop the raw (non-log) skewed columns since we use their log versions instead
df_model = df.drop(columns=['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome',
                             'LoanAmount', 'TotalIncome'])

print("=" * 70)
print("STEP 6: FINAL CLEANED DATASET (READY FOR MODELING)")
print("=" * 70)
print(f"Final shape: {df_model.shape}")
print(f"Final columns: {list(df_model.columns)}")
print(df_model.dtypes)
print(df_model.head())

# Save the fully cleaned dataset
df_model.to_csv('loan_dataset_cleaned.csv', index=False)
print("\nSaved cleaned dataset -> loan_dataset_cleaned.csv\n")

# ---------------------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X = df_model.drop(columns=['Loan_Status'])
y = df_model['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 70)
print("STEP 7: TRAIN/TEST SPLIT")
print("=" * 70)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train approval rate: {y_train.mean():.3f}, Test approval rate: {y_test.mean():.3f}\n")

# ---------------------------------------------------------------
# 8. FEATURE SCALING
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 9. TRAIN LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("=" * 70)
print("STEP 8-9: LOGISTIC REGRESSION MODEL TRAINED")
print("=" * 70)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)'])

print(f"Accuracy:  {acc:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{report}")

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("Feature Importance (Logistic Regression Coefficients):")
print(coef_df.to_string(index=False))

# Save results to a text file for the report
with open('model_results.txt', 'w') as f:
    f.write("LOAN APPROVAL PREDICTION - LOGISTIC REGRESSION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}\n\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"ROC-AUC:   {auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Feature Importance (Coefficients):\n")
    f.write(coef_df.to_string(index=False))

print("\nSaved full results -> model_results.txt")
