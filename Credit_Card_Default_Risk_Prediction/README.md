# Credit Card Default Risk Prediction

Predicts whether a credit card customer will default on their next payment,
using 30,000 records from the UCI dataset and eight tuned classifiers.

## Objective
Given a customer's payment history, credit limit, and demographic details,
predict the likelihood of default — a class-imbalanced problem where
defaulters are the minority.

## Approach
CRISP-DM workflow: data cleaning, feature scaling, train/test split with
cross-validation, and training of 8 classifiers with hyperparameter tuning.

## Models
Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient
Boosting, XGBoost and a tuned ensemble.

## Results
The best model reached **~81% test accuracy**. However, the more informative
story is class imbalance: while overall accuracy is high, recall on actual
defaulters is low (~0.26) and AUC ~0.61, because defaulters are a minority
class. This highlights why accuracy alone is misleading on imbalanced data —
precision, recall and AUC matter more for a risk-prediction use case.

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn.

## Dataset
UCI "Default of Credit Card Clients" dataset (30,000 records).
