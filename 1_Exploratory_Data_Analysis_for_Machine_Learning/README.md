# Movie Dataset Exploratory Data Analysis

An exploratory data analysis project examining movie popularity trends, voting patterns, and release strategies using the Latest Movie Dataset 2025 from Kaggle.

## Project Overview

This project performs exploratory data analysis on a dataset of 10,000 movies to uncover insights about movie popularity, voting behavior, and optimal release timing through data cleaning, feature engineering, visualization, and statistical hypothesis testing.

Dataset: 10,000 movies with 7 features (title, release_date, original_language, popularity, vote_count, vote_average, overview)  
Analysis: Data cleaning, feature engineering, visualizations, hypothesis testing  
Significance Level: α = 0.05

## Approach

### 1. Data Cleaning
   * Converted release_date to datetime format
   * Extracted Year and Month features from release dates
   * Removed rows with missing values (291 rows, 2.9% of data)
   * Dropped movies with zero votes (1,272 rows)
   * Removed extreme outliers using z-score thresholding (|z| > 3)

### 2. Feature Engineering
   * Created Year and Month columns for temporal analysis
   * Applied log transformations to popularity and vote_count for better distributions
   * One-hot encoded original_language for machine learning compatibility

### 3. Exploratory Visualizations
   * Distribution plots for popularity, vote count, and vote average (raw and log-transformed)
   * Time series analysis showing popularity trends from 1897 to 2025
   * Bar charts for monthly release patterns and top 10 languages

### 4. Hypothesis Testing

**Hypothesis 1: Post-2000 vs Pre-2000 Popularity**
   * Test: One-tailed t-test
   * Results: t = 18.48, p = 4.86 × 10⁻⁷⁵
   * Conclusion: Reject H₀ - Post-2000 movies are significantly more popular

**Hypothesis 2: English vs Non-English Vote Average**
   * Test: One-tailed t-test
   * Results: t = 1.49, p = 0.068
   * Conclusion: Fail to reject H₀ - No significant difference in vote averages

**Hypothesis 3: Popularity Across Release Months**
   * Test: One-way ANOVA
   * Results: F = 7.90, p = 7.90 × 10⁻⁷
   * Conclusion: Reject H₀ - Significant differences exist across months

## Key Findings

* Movies released after 2000 are significantly more popular than pre-2000 releases (p < 0.001)
* Release month has a statistically significant impact on movie popularity (p < 0.001)
* English movies do not receive significantly higher vote averages than non-English movies (p = 0.068)
* Movie popularity has increased steadily over time with a notable spike around 2020
* Summer months see marginally higher movie release volumes

## Tools & Libraries

* Data Processing: pandas, numpy
* Visualization: matplotlib, seaborn
* Statistical Testing: scipy.stats