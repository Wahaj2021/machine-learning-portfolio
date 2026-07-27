# Boston Neighbourhood House Price Prediction

A regression project predicting median house prices across Boston
neighbourhoods, comparing eight regression models and examining overfitting.

## Objective
Predict the median value of owner-occupied homes from neighbourhood
characteristics (crime rate, average rooms, accessibility, tax rate, etc.),
and compare how different regression algorithms generalise.

## Approach
Built following the CRISP-DM workflow:
- Exploratory analysis of feature distributions and correlations
- Cleaning, feature scaling, and a log transformation of the target
- Train/test split with cross-validation
- Training and tuning 8 regression models
- Comparison on train, cross-validation and test R² to detect overfitting

## Models compared
Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression,
Random Forest, Gradient Boosting and XGBoost.

## Results
Eight regression models were compared on cross-validation and held-out test R².
The best-generalising model reached a **test R² of ~0.80** (train R² 0.93),
predicting median home values well without severe overfitting. Notably, an
unregularised high-complexity model achieved a perfect training R² of 1.0 but
only 0.39 on test data — a clear example of overfitting, which the regularised
and ensemble models avoided.

## Tools
Python, pandas, scikit-learn, XGBoost, statsmodels, matplotlib, seaborn.

## Dataset
Boston Housing dataset (UCI Machine Learning Repository).
