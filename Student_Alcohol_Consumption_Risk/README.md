# Student Alcohol Consumption Risk Classification

Classifies students into Low, Medium or High alcohol-consumption risk using
demographic, family and academic features with eight tuned classifiers.

## Objective
Predict a student's alcohol-consumption risk band from features such as
family background, study time, grades and social activity.

## Approach
CRISP-DM workflow: cleaning, encoding categorical features, train/test split
with cross-validation, and training of 8 classifiers with tuning.

## Results
The best model reached **~73–75% test accuracy**. Performance is limited by
class imbalance — the "Medium" risk group dominates, so minority classes
(Low and High) are harder to predict, reflected in lower per-class recall.
This is a realistic outcome for a noisy behavioural dataset and a good
illustration of why macro-averaged metrics matter alongside accuracy.

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn.

## Dataset
Student Alcohol Consumption dataset (UCI).
