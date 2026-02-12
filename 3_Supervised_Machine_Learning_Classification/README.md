# Bank Term Deposit Predictor

A binary classification project to predict customer subscription to bank term deposits using the Bank Marketing Dataset from Kaggle.

## Project Overview

This project builds and compares multiple classification models to predict whether a customer will subscribe to a term deposit scheme. The analysis includes extensive feature engineering, class imbalance handling, hyperparameter optimization, and model interpretability using SHAP values.

Dataset: 45,211 customers with 16 features (age, job, marital status, education, balance, contact history, etc.)  
Target: Binary subscription outcome (heavily imbalanced: 88.3% no, 11.7% yes)  
Metric: ROC-AUC score

## Approach

### 1. Exploratory Data Analysis
   * Analyzed distributions of all 17 variables
   * Identified no missing values in the dataset
   * Discovered severe class imbalance (39,922 non-subscribers vs 5,289 subscribers)
   * Identified features requiring transformation (balance, age, duration, campaign, pdays)

### 2. Feature Engineering
   * Converted education and month to ordinal numeric values
   * Applied QuantileTransformer to balance feature (tested 9 transformation methods)
   * Log-transformed age and duration for better distribution
   * Binned campaign feature into 5 categories (1, 2-3, 4-5, 6-10, 10+)
   * Created pdays_cat from pdays with bins (<3mo, 3-6mo, 6-12mo, >12mo, no_contact)
   * Converted previous contacts to binary ever_contacted feature
   * One-hot encoded categorical variables (job, marital, poutcome, contact)
   * Standardized numerical features using StandardScaler

### 3. Model Development & Optimization
   * Algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost
   * Class Imbalance Techniques: SMOTE, Random Oversampling, Random Undersampling, class weights
   * Hyperparameter Tuning: Optuna with 50 trials per model (200 total trials)
   * Validation: 5-fold stratified cross-validation
   * Feature Importance: SHAP analysis on best model

## Results

Best Model: XGBoost (no sampling)  
Final ROC-AUC: 0.9341

| Model | Best ROC-AUC | Best Sampling Strategy |
|-------|--------------|------------------------|
| XGBoost | 0.9341 | None |
| Random Forest | 0.9256 | None |
| Decision Tree | 0.8975 | SMOTE |
| Logistic Regression | 0.8945 | Oversampling |

### Sampling Strategy Performance (XGBoost)
| Strategy | Mean ROC-AUC | Std Dev |
|----------|--------------|---------|
| None | 0.9311 | 0.0023 |
| Undersampling | 0.9264 | 0.0019 |
| SMOTE | 0.9262 | 0.0034 |
| Oversampling | 0.9258 | 0.0044 |

## Key Findings

* XGBoost achieved highest performance without requiring sampling techniques for class imbalance
* Top predictive features (from SHAP analysis): call duration, contact month, age, balance, previous outcome
* Logistic Regression produced more Type II errors (missed subscribers) but fewer Type I errors
* Random Forest generated more Type I errors (false positives)
* XGBoost balanced both error types while maintaining highest accuracy
* QuantileTransformer was most effective for normalizing heavily skewed balance distribution
* No sampling technique outperformed using class imbalance weights for tree-based models

## Model Recommendation

XGBoost is recommended because it:
* Minimizes Type I errors (reduces wasted outreach to unlikely subscribers)
* Maintains low Type II errors (captures most actual subscribers)
* Achieves best overall discrimination (highest ROC-AUC)
* Requires less manual intervention for class imbalance

## Next Steps

* Deploy model to production environment
* Implement continuous monitoring for model drift
* Track changes in customer contact patterns and adjust retraining schedule
* A/B test model-guided outreach strategy vs. current approach
* Collect additional features (transaction history, product ownership) to improve predictions
* Evaluate model fairness across demographic groups

## Tools & Libraries

* ML Frameworks: scikit-learn, XGBoost
* Imbalanced Learning: imbalanced-learn (SMOTE, resampling)
* Optimization: Optuna
* Interpretability: SHAP
* Data Processing: pandas, numpy, scipy
* Visualization: matplotlib, seaborn