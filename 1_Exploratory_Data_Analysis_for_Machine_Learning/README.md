\# Movie Dataset Exploratory Data Analysis



An exploratory data analysis project examining movie popularity trends, voting patterns, and release strategies using the Latest Movie Dataset 2025 from Kaggle.



\## Project Overview



This project performs exploratory data analysis on a dataset of 10,000 movies to uncover insights about movie popularity, voting behavior, and optimal release timing through data cleaning, feature engineering, visualization, and statistical hypothesis testing.



Dataset: 10,000 movies with 7 features (title, release\_date, original\_language, popularity, vote\_count, vote\_average, overview)  

Analysis: Data cleaning, feature engineering, visualizations, hypothesis testing  

Significance Level: α = 0.05



\## Approach



\### 1. Data Cleaning

&nbsp;  \* Converted release\_date to datetime format

&nbsp;  \* Extracted Year and Month features from release dates

&nbsp;  \* Removed rows with missing values (291 rows, 2.9% of data)

&nbsp;  \* Dropped movies with zero votes (1,272 rows)

&nbsp;  \* Removed extreme outliers using z-score thresholding (|z| > 3)



\### 2. Feature Engineering

&nbsp;  \* Created Year and Month columns for temporal analysis

&nbsp;  \* Applied log transformations to popularity and vote\_count for better distributions

&nbsp;  \* One-hot encoded original\_language for machine learning compatibility



\### 3. Exploratory Visualizations

&nbsp;  \* Distribution plots for popularity, vote count, and vote average (raw and log-transformed)

&nbsp;  \* Time series analysis showing popularity trends from 1897 to 2025

&nbsp;  \* Bar charts for monthly release patterns and top 10 languages



\### 4. Hypothesis Testing



\*\*Hypothesis 1: Post-2000 vs Pre-2000 Popularity\*\*

&nbsp;  \* Test: One-tailed t-test

&nbsp;  \* Results: t = 18.48, p = 4.86 × 10⁻⁷⁵

&nbsp;  \* Conclusion: Reject H₀ - Post-2000 movies are significantly more popular



\*\*Hypothesis 2: English vs Non-English Vote Average\*\*

&nbsp;  \* Test: One-tailed t-test

&nbsp;  \* Results: t = 1.49, p = 0.068

&nbsp;  \* Conclusion: Fail to reject H₀ - No significant difference in vote averages



\*\*Hypothesis 3: Popularity Across Release Months\*\*

&nbsp;  \* Test: One-way ANOVA

&nbsp;  \* Results: F = 7.90, p = 7.90 × 10⁻⁷

&nbsp;  \* Conclusion: Reject H₀ - Significant differences exist across months



\## Key Findings



\* Movies released after 2000 are significantly more popular than pre-2000 releases (p < 0.001)

\* Release month has a statistically significant impact on movie popularity (p < 0.001)

\* English movies do not receive significantly higher vote averages than non-English movies (p = 0.068)

\* Movie popularity has increased steadily over time with a notable spike around 2020

\* Summer months see marginally higher movie release volumes



\## Tools \& Libraries



\* Data Processing: pandas, numpy

\* Visualization: matplotlib, seaborn

\* Statistical Testing: scipy.stats

