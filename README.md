 Loan Approval Prediction using Machine Learning

## 📌 Project Overview

This project is an end-to-end Machine Learning solution for predicting whether a loan application is likely to be approved or rejected based on applicant and financial information.

The project covers the complete ML workflow, including data cleaning, exploratory analysis, feature engineering, categorical encoding, feature scaling, Logistic Regression model training, model evaluation, and an interactive web application.

## 🎯 Problem Statement

Loan approval decisions depend on multiple factors such as applicant income, co-applicant income, credit history, education, employment status, loan amount, loan term, and property area.

The goal of this project is to use Machine Learning to analyze these applicant characteristics and predict the loan approval status.

## 📊 Dataset

The project uses a publicly available loan approval dataset containing applicant information and loan approval outcomes.

The target variable is:

- Loan_Status
  - Y = Approved
  - N = Rejected

Important features include:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

## 🧹 Data Cleaning

The dataset was cleaned before training the Machine Learning model.

The preprocessing pipeline includes:

- Checking and removing duplicate records
- Removing duplicate Loan IDs
- Handling missing categorical values using the mode
- Handling missing loan amounts using the median
- Converting incorrect data types
- Converting 3+ dependents into a numerical value
- Standardizing categorical values

## ⚙️ Feature Engineering

Additional features were created to improve the model:

### Total Income

Applicant and co-applicant income were combined into a new TotalIncome feature.

### Log Transformation

Log transformations were applied to skewed financial features:

- LoanAmount_log
- TotalIncome_log

These transformations help make the financial variables more suitable for Logistic Regression.

## 🔢 Feature Encoding

Categorical variables were converted into numerical values.

Binary variables such as:

- Gender
- Married
- Education
- Self_Employed

were encoded into 0/1 values.

Property_Area was converted using one-hot encoding.

The target variable was also encoded:

- Approved → 1
- Rejected → 0

## 🤖 Machine Learning Model

The project uses *Logistic Regression* as the classification algorithm.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The split uses stratification to maintain a similar approval/rejection distribution between the training and testing sets.

Before training, the features are standardized using StandardScaler.

## 📈 Model Evaluation

The trained model is evaluated using:

- Accuracy
- ROC-AUC Score
- Confusion Matrix
- Classification Report

Logistic Regression coefficients are also analyzed to understand the influence of different features on loan approval predictions.

## 🌐 Web Application

An interactive web application was developed to allow users to enter applicant information and receive a predicted loan approval result.

The application provides a user-friendly interface for demonstrating the trained Machine Learning solution.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- HTML
- Machine Learning

## 📁 Project Files

- loan_dataset.csv — Original loan dataset
- loan_pipeline.py — Data cleaning, preprocessing, training, and evaluation pipeline
- loan_approval_app.html — Interactive web application
- loan_dataset_report.pdf — Dataset exploration report
- loan_data_cleaning_report.pdf — Data cleaning report
- loan_eda_report.pdf — Exploratory Data Analysis report
- phase5_logistic_regression_report.pdf — Logistic Regression report
- README.md — Project documentation

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Duplicate Removal
     ↓
Missing Value Handling
     ↓
Data Type Correction
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Loan Approval Prediction
     ↓
Web Application
