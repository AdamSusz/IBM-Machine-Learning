# Sentiment Analysis Using Deep Learning

A binary sentiment classification project comparing traditional machine learning and deep learning approaches on combined IMDB movie reviews and Twitter sentiment datasets.

## Project Overview

This project builds and evaluates three sentiment analysis models (Logistic Regression, Linear SVC, Bidirectional LSTM) to predict positive or negative sentiment from text. The models are trained on a combined dataset of movie reviews and tweets to improve generalization across different text contexts and writing styles.

Dataset: 83,374 samples from IMDB (40K) and Twitter (43K) combined  
Target: Binary sentiment (0=negative, 1=positive)  
Metric: Accuracy  
Class Distribution: Nearly balanced (51% negative, 49% positive)

## Approach

### 1. Data Preparation
   * Combined IMDB movie reviews (40K samples) with Twitter sentiment data (74K samples)
   * Dropped Twitter metadata columns (context, tweet_id)
   * Filtered Twitter data to binary labels (removed neutral and irrelevant entries)
   * Final dataset: 83,374 samples with balanced classes (42,561 negative, 40,813 positive)

### 2. Text Preprocessing
   * Removed HTML tags from text
   * Stripped punctuation and numbers, keeping only alphabetic characters
   * Converted text to lowercase
   * Removed stopwords except negations (no, not, nor, never, ain't)
   * Applied lemmatization using WordNetLemmatizer for normalization
   * Tokenized preprocessed text

### 3. Exploratory Analysis
   * Generated TF-IDF weighted word clouds for positive and negative sentiments
   * Visualized distinctive vocabulary between sentiment classes
   * Identified key sentiment-bearing words beyond generic stopwords

### 4. Model Development

**Traditional Models:**
   * TF-IDF vectorization with max 20,000 features and bigrams
   * Logistic Regression baseline
   * Linear SVC for comparison

**Deep Learning Model:**
   * Text vectorization layer (max 20K tokens, sequence length 200)
   * Embedding layer (20 dimensions)
   * Spatial dropout (0.2) for regularization
   * LSTM layer (128 units, return sequences)
   * Dropout (0.3)
   * Bidirectional LSTM layer (64 units)
   * Dropout (0.3)
   * Dense layer (32 units, ReLU)
   * Dropout (0.2)
   * Output layer (1 unit, sigmoid)
   * Optimizer: Adam with learning_rate=0.001
   * Loss: Binary crossentropy
   * Training: 10 epochs, batch size 32

## Results

Best Performance: Bidirectional LSTM  
Test Accuracy: 88.3%

| Model | Test Accuracy | Training Time |
|-------|---------------|---------------|
| Bidirectional LSTM | **88.3%** | 1h 41m 55s |
| Linear SVC | 87.6% | ~10s |
| Logistic Regression | 86.4% | ~10s |

### Training Progress (LSTM)

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------------|-----------------|
| 1 | 78.8% | 85.8% | 0.3487 |
| 2 | 88.1% | 87.0% | 0.3328 |
| 3 | 90.5% | 87.4% | 0.3154 |
| 4 | 92.1% | 87.9% | 0.3126 |
| 5 | 93.4% | 88.3% | 0.3323 |
| 6-10 | 94.4-96.4% | 88.1-88.6% | 0.34-0.40 |

**Overfitting observed:** Training accuracy reached 96.4% while validation plateaued at ~88.3% after epoch 5

## Key Findings

* Bidirectional LSTM achieved highest accuracy (88.3%) with 1.9% improvement over Linear SVC
* Traditional models trained 600× faster (~10s vs ~2h) with only marginal accuracy loss
* LSTM overfitting began after epoch 5 despite multiple dropout layers (0.2-0.3)
* TF-IDF with bigrams provided strong baseline performance for traditional models
* Combined IMDB + Twitter datasets improved model generalization across contexts
* Prediction inference remains fast (<1s) regardless of training time

## Model Deployment Features

* `predict_sentiment()` function for real-time inference
* Returns sentiment label (Positive/Negative) with confidence percentage
* Outputs probability distribution across both classes
* Example: "I went to the movies and saw the worst movie imaginable" → Negative (97.73% confidence)

## Trade-offs & Recommendations

**Use LSTM when:**
* Maximum accuracy is critical (production systems, high-stakes applications)
* Training time is not a constraint (offline training acceptable)
* Computational resources available for inference at scale

**Use Linear SVC when:**
* Fast training/iteration required (experimentation, rapid prototyping)
* Near-optimal performance acceptable (1.9% accuracy trade-off)
* Limited computational resources
* Simpler model interpretability preferred

## Future Improvements

* Implement early stopping to prevent overfitting (stop at epoch 5)
* Hyperparameter tuning for traditional models (C parameter, regularization)
* Add more training data to improve generalization
* Deploy model with continuous monitoring for performance drift
* A/B test model versions in production

## Tools & Libraries

* Deep Learning: TensorFlow/Keras (LSTM, Bidirectional, Embedding)
* Traditional ML: scikit-learn (LogisticRegression, LinearSVC, TfidfVectorizer)
* NLP Processing: NLTK (stopwords, tokenization, lemmatization)
* Data Processing: pandas, numpy
* Visualization: matplotlib, seaborn, WordCloud