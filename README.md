# Machine Learning Portfolio

Five end-to-end machine learning projects in Python — classification and
regression across finance, healthcare and education. Each follows the CRISP-DM
workflow: data understanding, cleaning, feature preparation, model training,
hyperparameter tuning and evaluation.

## Projects

| Project | Type | Result | Key takeaway |
|---------|------|--------|--------------|
| [Credit Card Default](Credit_Card_Default_Risk_Prediction) | Classification | ~81% accuracy | High accuracy hides class imbalance — defaulter recall and AUC matter more |
| [Boston House Prices](Boston_House_Price_Prediction) | Regression | ~0.80 test R² | Regularised and ensemble models avoided the overfitting seen in complex models |
| [Disease Diagnosis](Symptom_Based_Disease_Diagnosis) | Multi-class (42) | 100% accuracy | Perfect scores reflect clean, deterministic data — not model superiority |
| [Student Alcohol Risk](Student_Alcohol_Consumption_Risk) | Classification | ~73–75% accuracy | Realistic noisy data; class imbalance limits minority-class prediction |
| [Iris Species](Iris_Flower_Species_Identification) | Classification | 100% accuracy | Clean baseline demonstrating the full workflow on an easy, well-separated set |

## Methods and tools

- **Libraries:** Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn
- **Techniques:** train/test split, cross-validation, GridSearchCV tuning,
  feature scaling, handling class imbalance
- **Models:** Logistic Regression, Decision Trees, Random Forest, SVM, KNN,
  Gradient Boosting, XGBoost, and regularised regression (Ridge, Lasso, ElasticNet)
- **Evaluation:** accuracy, precision/recall, F1, ROC-AUC (classification);
  RMSE, R² (regression)
- **Workflow:** CRISP-DM across all projects

## Note

Datasets are sourced from Kaggle and the UCI Machine Learning Repository.
Notebooks render directly in-browser on GitHub, including all charts and outputs.
