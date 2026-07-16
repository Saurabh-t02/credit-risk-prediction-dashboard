# Credit Risk Prediction Dashboard

An end-to-end machine learning project that predicts loan default risk and 
visualizes results through an interactive Streamlit dashboard.

## Live Demo
[View the live dashboard]([your-streamlit-url-here](https://credit-risk-prediction-dashboard-dj8ll9ezjhnc4zse8jn6nr.streamlit.app/))

## Overview
This project trains and compares two classification models on the 
["Give Me Some Credit"](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) 
dataset (150,000 loan applicants) to predict the probability that a borrower 
will default within two years.

## Tech Stack
- **Python** — pandas, NumPy for data cleaning and feature engineering
- **Scikit-learn** — Logistic Regression & Random Forest classifiers
- **Streamlit** — interactive dashboard and real-time prediction interface
- **Matplotlib/Seaborn** — EDA and feature importance visualization

## Approach
1. Cleaned missing values in `MonthlyIncome` and `NumberOfDependents`, capped outliers in key financial ratios.
2. Performed EDA to understand class imbalance (~6.7% default rate) and feature correlations.
3. Trained Logistic Regression (baseline) and Random Forest models with `class_weight="balanced"` to address the imbalance.
4. Evaluated using Precision, Recall, and AUC-ROC (accuracy alone is misleading on imbalanced data).
5. Built a Streamlit dashboard for feature-importance visualization and real-time applicant risk scoring.

## Results
| Model | AUC-ROC | Precision | Recall |
|---|---|---|---|
| Logistic Regression | your number | your number | your number |
| Random Forest | your number | your number | your number |

## Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Dataset
[Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) (Kaggle)
