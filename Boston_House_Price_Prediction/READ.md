# Boston Neighbourhood House Price Prediction

A regression project predicting median house prices across Boston
neighbourhoods, comparing eight regression models to find the best performer.

## Objective
Predict the median value of owner-occupied homes from neighbourhood
characteristics (crime rate, number of rooms, accessibility, tax rate, etc.),
and compare how different regression algorithms perform on the task.

## Approach
Built following the CRISP-DM workflow:
- Data understanding and exploratory analysis of the feature distributions
- Cleaning, feature scaling and train/test splitting
- Training and tuning 8 regression models
- Evaluation and comparison on held-out test data

## Models compared
Linear Regression, Ridge, Lasso, Polynomial Regression, Decision Tree,
Random Forest, Gradient Boosting and XGBoost.

## Evaluation
Models compared on RMSE and R² on the test set to identify the strongest
predictor and assess overfitting.

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn.

## Dataset
Boston Housing dataset (originally from the UCI Machine Learning Repository).
