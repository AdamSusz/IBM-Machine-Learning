# Insurance Cost Analysis

A regression analysis project for predicting medical insurance charges based on personal and lifestyle factors using the Medical Cost Personal Dataset from Kaggle.

## Project Overview

This project builds and compares multiple regression models to predict insurance charges based on demographic and health-related features. The analysis includes exploratory data visualization, feature importance analysis, and performance comparison across six different regression algorithms.

Dataset: 1,338 samples with 6 features (age, sex, bmi, children, smoker, region)  
Target: Medical insurance charges (continuous)  
Metric: Root Mean Squared Error (RMSE)

## Approach

### 1. Exploratory Data Analysis
   * Analyzed distributions of age, BMI, number of children, smoking status, region, and sex
   * Examined relationships between features and insurance charges through scatter plots and box plots
   * Identified smoker status, age, and BMI as key predictors
   * Applied log transformation to charges for better visualization

### 2. Data Preprocessing
   * Standardized numerical features (age, bmi, children)
   * One-hot encoded categorical variables (sex, smoker, region)
   * Applied 70-30 train-test split
   * Used ColumnTransformer pipeline for preprocessing

### 3. Model Development
   * Algorithms: Linear Regression, Ridge, Lasso, Elastic Net, XGBoost, Random Forest
   * Validation: 5-fold cross-validation on training set
   * Feature Importance: Extracted from Random Forest model to identify key predictors

## Results

Final Test RMSE: 4,338.20

| Model | Validation RMSE |
|-------|-----------------|
| Random Forest | 4,658.58 |
| XGBoost | 4,965.79 |
| Elastic Net | 6,200.43 |
| Ridge Regression | 6,200.91 |
| Linear Regression | 6,201.61 |
| Lasso Regression | 6,201.61 |

## Key Findings

* Random Forest Regressor significantly outperformed linear models (25% improvement)
* Linear regression variants showed similar performance, with regularization providing minimal benefit
* Feature importance analysis revealed smoker status, BMI, and age as the strongest predictors
* Smokers pay substantially higher insurance charges than non-smokers
* Insurance costs increase with age and BMI
* High variance in target variable (std ≈ 12,000) makes prediction challenging with current features

## Limitations & Future Work

* Small dataset size (1,338 samples) limits model generalization
* Only 6 features available - additional data could improve predictions
* High target variance (RMSE of 4,338 relative to std of 12,000)
* Potential improvements: feature engineering (age-BMI-smoking interactions), external data sources, ensemble methods

## Tools & Libraries

* ML Frameworks: scikit-learn, XGBoost
* Data Processing: pandas, numpy
* Visualization: matplotlib, seaborn