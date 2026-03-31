# Credit Card Fraud Detection

Real-world credit card fraud detection using machine learning on an imbalanced dataset.  
284,807 transactions, 0.17% fraud — from European cardholders, September 2013.

## Overview
The core challenge is extreme class imbalance. A model predicting "not fraud" every time 
achieves 99.8% accuracy — making accuracy a useless metric. This project focuses on 
Precision, Recall, and F1-Score to evaluate real fraud detection performance.

## Dataset
[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle  
Features V1–V28 are PCA-transformed to protect cardholder privacy. Only `Amount` and `Time` are raw.

## Models
| Model | Precision | Recall | F1 | False Alarms | Missed Frauds |
|---|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 1,384 | 8 |
| Random Forest (0.50) | 0.96 | 0.74 | 0.84 | 3 | 25 |
| Random Forest (0.30) | 0.92 | 0.85 | 0.88 | 7 | 15 |
| XGBoost (0.50) | 0.89 | 0.83 | 0.86 | 10 | 17 |
| **XGBoost (0.98)** | **0.98** | **0.81** | **0.88** | **2** | **19** |

**Best model: XGBoost with threshold tuning at 0.98**  
Near-perfect precision with only 2 false alarms across 56,864 legitimate transactions.

## Key Findings
- Fraud clusters at very low amounts (€1–€10), consistent with card testing behavior
- V14 is the strongest fraud signal — ranked #1 by both Random Forest and XGBoost
- Threshold tuning is critical: shifting from 0.50 to 0.98 reduced false alarms from 10 to 2
- Linear correlation alone is a poor judge of feature importance in fraud detection

## Stack
Python · Pandas · Scikit-learn · XGBoost · Matplotlib · Seaborn