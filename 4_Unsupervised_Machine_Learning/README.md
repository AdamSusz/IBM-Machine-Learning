# Fashion-MNIST Unsupervised Clustering

An unsupervised learning project comparing clustering algorithms and dimensionality reduction techniques on the Fashion-MNIST dataset from Kaggle.

## Project Overview

This project evaluates three clustering algorithms (K-Means, Gaussian Mixture Models, Agglomerative Clustering) combined with four dimensionality reduction methods (PCA, Autoencoder, UMAP, None) to cluster 28×28 grayscale fashion images into 10 distinct categories without using labels during training.

Dataset: 60,000 training images with 784 features (28×28 pixels)  
Classes: 10 fashion items (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)  
Metrics: Silhouette Score, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)

## Approach

### 1. Data Preparation
   * Normalized pixel values from 0-255 to 0-1 range
   * Visualized sample images from each of 10 classes
   * For Agglomerative Clustering: used stratified subset of 20,000 samples due to memory constraints

### 2. Dimensionality Reduction Methods
   * **PCA 90%**: Reduced to 83 components retaining 90% variance
   * **PCA 95%**: Reduced to 186 components retaining 95% variance
   * **Autoencoder**: Neural network with 512→256→128→64 latent dimensions, trained for 20 epochs
   * **UMAP**: Reduced to 32 components with n_neighbors=15, min_dist=0.1
   * **No Reduction**: Full 784-dimensional feature space (K-Means baseline only)

### 3. Clustering Algorithms
   * **K-Means**: 10 clusters, k-means++ initialization, tested on all reduction methods
   * **Gaussian Mixture**: 10 components, full covariance, k-means++ initialization, tested on reduced data
   * **Agglomerative**: 10 clusters, Ward linkage, tested on 20K subset due to O(n²) memory

### 4. Evaluation
   * Silhouette Score: measures cluster separation and cohesion
   * ARI: agreement with ground truth labels (corrected for chance)
   * NMI: information shared between predictions and true labels
   * Runtime analysis for each method combination

## Results

Best Performance: K-Means + UMAP  
Silhouette: 0.5099 | ARI: 0.4873 | NMI: 0.6489 | Time: 14s (+ 170s UMAP)

### K-Means Performance

| Method | Silhouette | ARI | NMI | Clustering Time (s) |
|--------|------------|-----|-----|---------------------|
| UMAP | **0.5099** | **0.4873** | **0.6489** | 14 |
| PCA 90% | 0.1730 | 0.3750 | 0.5131 | 21.8 |
| PCA 95% | 0.1679 | 0.3489 | 0.5128 | 40.2 |
| Autoencoder | 0.1564 | 0.2840 | 0.4847 | 4.17 |
| No Reduction | 0.1433 | 0.3750 | 0.5131 | 189 |

### Gaussian Mixture Performance

| Method | Silhouette | ARI | NMI | Clustering Time (s) |
|--------|------------|-----|-----|---------------------|
| UMAP | **0.4701** | **0.4868** | **0.6463** | 9.55 |
| Autoencoder | 0.0798 | 0.4293 | 0.5864 | 165 |
| PCA 95% | 0.0663 | 0.3673 | 0.5515 | 1,406 |
| PCA 90% | 0.0640 | 0.3181 | 0.5144 | 277 |

### Agglomerative Clustering Performance (20K subset)

| Method | Silhouette | ARI | NMI | Clustering Time (s) |
|--------|------------|-----|-----|---------------------|
| PCA 90% | **0.1483** | **0.4311** | **0.5928** | 31.2 |
| PCA 95% | 0.1343 | 0.4311 | 0.5928 | 30.3 |
| UMAP | 0.3086 | 0.4048 | 0.5508 | 166 |
| Autoencoder | 0.1131 | 0.3185 | 0.5271 | 26.9 |

### Dimensionality Reduction Runtimes (Full Dataset)

| Method | Time (s) | Output Dimensions |
|--------|----------|-------------------|
| PCA 90% | 5.25 | 83 |
| PCA 95% | 5.83 | 186 |
| UMAP | 170 | 32 |
| Autoencoder | 342 | 64 |

## Key Findings

* **UMAP consistently outperformed other dimensionality reduction methods** across K-Means and Gaussian Mixture models
* UMAP achieved 3× better silhouette score (0.51 vs 0.17) compared to PCA for K-Means
* K-Means + UMAP provided best overall clustering quality with clear visual separation between clusters
* Gaussian Mixture Models struggled with image data, achieving lower silhouette scores (<0.08) with PCA
* Agglomerative Clustering performed best with PCA due to distance-preserving properties of linear transformations
* UMAP's nonlinear manifold learning effectively disentangled class structure despite distorting global distances
* Autoencoder representations underperformed, suggesting the latent space wasn't optimally suited for clustering
* Dimensionality reduction dramatically reduced runtime (189s → 14s for K-Means) while improving quality

## Challenges & Observations

* Low-resolution 28×28 images limit feature distinguishability between similar items (shirts vs pullovers, sneakers vs boots)
* Best silhouette score (0.51) indicates moderate cluster overlap - expected given visual similarity across classes
* Agglomerative Clustering memory constraints (O(n²)) required subset sampling, potentially affecting results
* UMAP's computational cost (170s) offset by significant clustering quality improvements

## Future Improvements

* Reduce number of clusters to broader categories (footwear, tops, bottoms) for higher quality separation
* Use higher resolution images to capture finer visual details
* Test semi-supervised approaches with small labeled subset to guide clustering
* Explore ensemble clustering methods combining multiple algorithms
* Investigate deep clustering methods (DEC, IDEC) that jointly learn representations and clusters

## Tools & Libraries

* Clustering: scikit-learn (KMeans, GaussianMixture, AgglomerativeClustering)
* Dimensionality Reduction: scikit-learn (PCA), UMAP, TensorFlow/Keras (Autoencoder)
* Evaluation: scikit-learn metrics (silhouette_score, adjusted_rand_score, normalized_mutual_info_score)
* Data Processing: pandas, numpy
* Visualization: matplotlib, seaborn