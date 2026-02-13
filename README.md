# Machine Learning Portfolio

A comprehensive collection of machine learning projects spanning exploratory data analysis, supervised learning (regression and classification), unsupervised learning, and deep learning. Each project demonstrates end-to-end workflows including data preprocessing, model development, evaluation, and analysis.

## Projects Overview

### 1. Exploratory Data Analysis for Machine Learning
Analyzed 10,000 movies from Kaggle to identify factors influencing popularity through statistical hypothesis testing. Performed data cleaning, feature engineering with log transformations, and conducted t-tests and ANOVA. Found that post-2000 movies are significantly more popular (p < 0.001) and release month impacts success.

**Tools:** pandas, matplotlib, seaborn, scipy.stats

---

### 2. Supervised Machine Learning: Regression
Predicted medical insurance charges for 1,338 patients using personal and health factors. Compared six regression models with Random Forest achieving best performance (RMSE: 4,338). Feature importance analysis revealed smoking status, BMI, and age as strongest predictors.

**Best Model:** Random Forest (RMSE: 4,338) | **Tools:** scikit-learn, XGBoost

---

### 3. Supervised Machine Learning: Classification
Built binary classifier to predict bank term deposit subscriptions for 45,211 customers. Tested multiple algorithms with extensive hyperparameter tuning via Optuna and class imbalance handling techniques. XGBoost achieved 93.4% ROC-AUC without requiring sampling methods. SHAP analysis identified call duration and contact patterns as key drivers.

**Best Model:** XGBoost (ROC-AUC: 0.9341) | **Tools:** scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn

---

### 4. Unsupervised Machine Learning
Clustered 60,000 Fashion-MNIST grayscale images into 10 categories without labels. Evaluated three clustering algorithms (K-Means, Gaussian Mixture, Agglomerative) with four dimensionality reduction techniques (PCA, Autoencoder, UMAP, None). UMAP dramatically outperformed other methods, achieving 3× better cluster separation.

**Best Approach:** K-Means + UMAP (Silhouette: 0.51) | **Tools:** scikit-learn, UMAP, TensorFlow/Keras

---

### 5. Deep Learning and Reinforcement Learning
Created sentiment classifier on 83,374 combined IMDB reviews and tweets. Compared traditional ML (Logistic Regression, SVM) against Bidirectional LSTM architecture. Deep learning achieved marginal improvement (1.9%) over Linear SVC with 600× longer training time, highlighting accuracy-efficiency trade-offs.

**Best Model:** Bidirectional LSTM (88.3% accuracy) | **Tools:** TensorFlow/Keras, scikit-learn, NLTK

---

## Skills Demonstrated

**Machine Learning:** Regression, Classification, Clustering, Deep Learning (LSTM)  
**Optimization:** Hyperparameter tuning, Cross-validation, Regularization  
**Data Science:** EDA, Statistical testing, Feature engineering, Model evaluation  
**Tools:** Python (scikit-learn, TensorFlow/Keras, XGBoost, pandas, NLTK)

## Learning Outcomes

* End-to-end machine learning workflows from data cleaning to model deployment
* Comparative analysis across algorithm families (linear, tree-based, neural networks)
* Advanced preprocessing techniques for structured data, text, and images
* Handling real-world challenges (missing data, outliers, class imbalance, overfitting)
* Model interpretability and evaluation without relying solely on accuracy metrics
* Trade-off analysis between model complexity, performance, and computational cost

## Future Directions

* Explore transformer architectures (BERT, GPT) for NLP tasks
* Implement ensemble methods combining multiple models
* Deploy models with MLOps pipelines (Docker, CI/CD)
* Experiment with reinforcement learning for sequential decision-making
* Add computer vision projects using CNNs and transfer learning