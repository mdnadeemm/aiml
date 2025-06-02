# Chapter 6: Clustering Methods for Data Analysis

## Learning Objectives
By the end of this chapter, you will be able to:
- Understand different clustering algorithms and their applications
- Implement various clustering techniques (K-means, Hierarchical, DBSCAN)
- Evaluate clustering performance using appropriate metrics
- Choose the right clustering method for different data types
- Handle challenges in clustering analysis

## What is Clustering?

Clustering is an unsupervised learning technique that groups similar data points together without predefined labels. The goal is to discover hidden patterns and structures in data.

### Key Characteristics:
- **Unsupervised**: No target variable or labels
- **Exploratory**: Discovers hidden patterns
- **Similarity-based**: Groups similar observations
- **Partitioning**: Divides data into meaningful groups

### Applications:
- Customer segmentation
- Gene sequencing
- Image segmentation
- Market research
- Social network analysis
- Anomaly detection

## K-Means Clustering

### Algorithm Steps:
1. Choose number of clusters (k)
2. Initialize cluster centroids randomly
3. Assign each point to nearest centroid
4. Update centroids as cluster means
5. Repeat until convergence

### Mathematical Foundation:
The objective of K-Means is to minimize the within-cluster sum of squares (WCSS), also known as inertia:
```
WCSS = Σ_{j=1}^{k} Σ_{x_i ∈ S_j} ||x_i - μ_j||²
```
Where:
- `k` is the number of clusters
- `S_j` is the set of points in cluster `j`
- `μ_j` is the centroid of cluster `j`
- `x_i` is a data point

### Implementation:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np # Import numpy for example data

# Example Data (replace with your actual data)
# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Create and fit model
# kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto') # n_init='auto' to suppress warning
# cluster_labels = kmeans.fit_predict(X)

# Visualize results (ensure X and cluster_labels are defined)
# plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='black')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, alpha=0.75)
# plt.title('K-Means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()
```

### Choosing Optimal K:
- **Elbow Method**: Plot WCSS for different values of `k`. Look for an "elbow" point where adding more clusters doesn't significantly reduce WCSS.
  ```python
  # wcss = []
  # for i in range(1, 11):
  #     kmeans_elbow = KMeans(n_clusters=i, random_state=42, n_init='auto')
  #     kmeans_elbow.fit(X) # Assuming X is defined
  #     wcss.append(kmeans_elbow.inertia_)
  # plt.plot(range(1, 11), wcss)
  # plt.title('Elbow Method')
  # plt.xlabel('Number of clusters (k)')
  # plt.ylabel('WCSS')
  # plt.show()
  ```
- **Silhouette Analysis**: Measures how similar a point is to its own cluster compared to other clusters. Values range from -1 to 1; higher values are better.
  ```python
  # from sklearn.metrics import silhouette_score
  # # For a given k, e.g., k=3:
  # # kmeans_silhouette = KMeans(n_clusters=3, random_state=42, n_init='auto')
  # # labels_silhouette = kmeans_silhouette.fit_predict(X) # Assuming X is defined
  # # silhouette_avg = silhouette_score(X, labels_silhouette)
  # # print(f"For n_clusters = 3, the average silhouette_score is : {silhouette_avg}")
  ```
- **Gap Statistic**: Compares the WCSS of your clustering to the WCSS of random data.

## Hierarchical Clustering

### Types:
1.  **Agglomerative (Bottom-up)**: Starts with each point as its own cluster and iteratively merges the closest pairs of clusters until only one cluster (or k clusters) remains.
2.  **Divisive (Top-down)**: Starts with all points in one cluster and recursively splits clusters until each point is its own cluster (or k clusters are formed). Agglomerative is more common.

### Linkage Criteria (for Agglomerative):
Determines how the distance between clusters is calculated:
- **Single Linkage**: Minimum distance between any two points in the two clusters.
- **Complete Linkage**: Maximum distance between any two points in the two clusters.
- **Average Linkage**: Average distance between all pairs of points (one from each cluster).
- **Ward's Linkage**: Minimizes the increase in total within-cluster variance after merging. Often performs well.

### Implementation:
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np # For example data

# Example Data
# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Hierarchical clustering using AgglomerativeClustering
# agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
# cluster_labels_agg = agg_clustering.fit_predict(X) # Assuming X is defined

# Create dendrogram to visualize hierarchy and help choose k
# linkage_matrix = linkage(X, method='ward') # method should match linkage in AgglomerativeClustering
# plt.figure(figsize=(10, 7))
# dendrogram(linkage_matrix)
# plt.title('Hierarchical Clustering Dendrogram (Ward)')
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()
```

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### Key Concepts:
- **Core points**: A point is a core point if it has at least `min_samples` other points within a distance of `eps`.
- **Border points**: A point is a border point if it is not a core point but is within `eps` distance of a core point.
- **Noise points**: A point that is neither a core point nor a border point. These points do not belong to any cluster.

### Parameters:
- **`eps` (epsilon)**: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- **`min_samples`**: The number of samples in a neighborhood for a point to be considered as a core point.

### Advantages:
- Can find clusters of arbitrary shapes.
- Robust to outliers (identifies them as noise).
- Does not require specifying the number of clusters beforehand (though `eps` and `min_samples` act as indirect controls).

### Implementation:
```python
from sklearn.cluster import DBSCAN
import numpy as np # For example data
import matplotlib.pyplot as plt

# Example Data
# X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]) # Added an outlier

# DBSCAN clustering
# dbscan = DBSCAN(eps=3, min_samples=2) # eps and min_samples need tuning
# cluster_labels_dbscan = dbscan.fit_predict(X) # Assuming X is defined

# Visualize results (ensure X and cluster_labels_dbscan are defined)
# unique_labels = set(cluster_labels_dbscan)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1: # Noise points
#         col = [0, 0, 0, 1] # Black for noise

#     class_member_mask = (cluster_labels_dbscan == k)
#     xy = X[class_member_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14 if k != -1 else 6)

# plt.title(f'DBSCAN Clustering (Estimated clusters: {len(unique_labels)- (1 if -1 in unique_labels else 0)})')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()
```

## Clustering Evaluation Metrics

### Internal Metrics (No ground truth labels needed):
These metrics evaluate the quality of the clustering structure itself.
#### Silhouette Score:
Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
- Range: -1 to +1.
- High value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
```python
from sklearn.metrics import silhouette_score
# score = silhouette_score(X, cluster_labels) # Assuming X and cluster_labels from a model are defined
# print(f"Silhouette Score: {score:.3f}")
```

#### Calinski-Harabasz Index (Variance Ratio Criterion):
Ratio of the sum of between-cluster dispersion and within-cluster dispersion.
- Higher values generally indicate better clustering (denser and well-separated clusters).
```python
from sklearn.metrics import calinski_harabasz_score
# score = calinski_harabasz_score(X, cluster_labels) # Assuming X and cluster_labels are defined
# print(f"Calinski-Harabasz Score: {score:.3f}")
```

#### Davies-Bouldin Index:
Measures the average similarity ratio of each cluster with its most similar cluster.
- Lower values indicate better clustering (clusters are more separated and less dispersed).
```python
from sklearn.metrics import davies_bouldin_score
# score = davies_bouldin_score(X, cluster_labels) # Assuming X and cluster_labels are defined
# print(f"Davies-Bouldin Score: {score:.3f}")
```

### External Metrics (Require ground truth labels):
These metrics compare the clustering results to a known ground truth.
#### Adjusted Rand Index (ARI):
Measures the similarity between true and predicted clusterings, adjusted for chance.
- Range: -1 to 1. Higher is better. 1 is perfect agreement.
```python
from sklearn.metrics import adjusted_rand_score
# score = adjusted_rand_score(true_labels, predicted_labels) # Assuming true_labels and predicted_labels are defined
# print(f"Adjusted Rand Index: {score:.3f}")
```

#### Normalized Mutual Information (NMI):
Measures the agreement of two assignments, ignoring permutations and normalized against chance.
- Range: 0 to 1. Higher is better.
```python
from sklearn.metrics import normalized_mutual_info_score
# score = normalized_mutual_info_score(true_labels, predicted_labels)
# print(f"Normalized Mutual Information: {score:.3f}")
```

## Advanced Clustering Techniques

### Gaussian Mixture Models (GMM):
Assumes data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
```python
from sklearn.mixture import GaussianMixture
# gmm = GaussianMixture(n_components=3, random_state=42) # n_components is number of clusters
# cluster_labels_gmm = gmm.fit_predict(X) # Assuming X is defined
# Probabilistic assignment:
# probabilities_gmm = gmm.predict_proba(X)
```

### Spectral Clustering:
Performs dimensionality reduction using eigenvalues of a similarity matrix before clustering in fewer dimensions. Good for non-globular clusters.
```python
from sklearn.cluster import SpectralClustering
# spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
# cluster_labels_spectral = spectral.fit_predict(X) # Assuming X is defined
```

### Mean Shift Clustering:
A centroid-based algorithm that aims to discover "blobs" in a smooth density of samples. Does not require specifying the number of clusters.
```python
from sklearn.cluster import MeanShift, estimate_bandwidth
# Bandwidth is a crucial parameter, can be estimated
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
# mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# cluster_labels_ms = mean_shift.fit_predict(X) # Assuming X is defined
```

## Practical Tips

### Data Preprocessing:
- **Feature Scaling**: Algorithms like K-Means that use Euclidean distance are sensitive to feature scales. Standardize or Normalize your data.
- **Handle Missing Values**: Impute or remove missing values.
- **Outliers**: Can significantly affect K-Means and Hierarchical clustering (Ward's). DBSCAN is robust to outliers. Consider outlier detection and removal if appropriate.
- **Dimensionality Reduction**: Can improve performance and interpretability if you have many features.

### Algorithm Selection:
- **K-Means**: Good for spherical clusters and when you have an idea of `k`. Computationally efficient.
- **Hierarchical Clustering**: Useful when you want to see a hierarchy of clusters (dendrogram). Does not require `k` beforehand but can be computationally expensive for large datasets.
- **DBSCAN**: Excellent for clusters of arbitrary shapes and when dealing with noise/outliers. Requires tuning of `eps` and `min_samples`.
- **GMM**: Good for overlapping clusters and provides probabilistic assignments.
- **Spectral Clustering**: Effective for complex, non-convex cluster shapes but can be sensitive to parameter choices.

### Validation:
- Use a combination of internal and external (if available) metrics.
- Visualize results (e.g., scatter plots, dendrograms) when possible, especially in 2D or 3D (after PCA).
- Consider domain knowledge to assess if the clusters make sense.
- Test stability: Do you get similar clusters if you resample your data or change initializations?

## Real-World Case Study: Customer Segmentation
Imagine a retail company wants to segment its customers based on their purchasing behavior (e.g., annual income, spending score).

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Example: Create dummy customer data
# customer_data = pd.DataFrame({
#     'Annual_Income_k': np.random.randint(15, 150, 200),
#     'Spending_Score_1_100': np.random.randint(1, 100, 200)
# })
# X_customers = customer_data[['Annual_Income_k', 'Spending_Score_1_100']].values

# Preprocess data
# scaler = StandardScaler()
# X_customers_scaled = scaler.fit_transform(X_customers)

# Find optimal number of clusters using Silhouette Score
# silhouette_scores_cs = []
# k_range_cs = range(2, 11) # Test k from 2 to 10

# for k_cs in k_range_cs:
#     kmeans_cs = KMeans(n_clusters=k_cs, random_state=42, n_init='auto')
#     cluster_labels_cs = kmeans_cs.fit_predict(X_customers_scaled)
#     silhouette_avg_cs = silhouette_score(X_customers_scaled, cluster_labels_cs)
#     silhouette_scores_cs.append(silhouette_avg_cs)
#     print(f"For n_clusters = {k_cs}, silhouette score is {silhouette_avg_cs:.3f}")

# plt.plot(k_range_cs, silhouette_scores_cs)
# plt.title('Silhouette Scores for Different k')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.show()

# Select best k (e.g., based on highest silhouette score) and create final model
# best_k_cs = k_range_cs[np.argmax(silhouette_scores_cs)] # Example: find k with max score
# print(f"Best k based on Silhouette Score: {best_k_cs}")
# final_kmeans_model = KMeans(n_clusters=best_k_cs, random_state=42, n_init='auto')
# customer_data['Segment'] = final_kmeans_model.fit_predict(X_customers_scaled)

# Analyze segments
# print("\nCustomer Segments Head:")
# print(customer_data.head())
# print("\nSegment Analysis (Counts):")
# print(customer_data['Segment'].value_counts())

# for i in range(best_k_cs):
#     segment_df = customer_data[customer_data['Segment'] == i]
#     print(f"\nSegment {i} Characteristics:")
#     print(f"  Number of Customers: {len(segment_df)}")
#     print(f"  Avg Annual Income: {segment_df['Annual_Income_k'].mean():.2f}k")
#     print(f"  Avg Spending Score: {segment_df['Spending_Score_1_100'].mean():.2f}")

# Visualize segments (if X_customers was used)
# plt.figure(figsize=(10, 6))
# unique_segments = customer_data['Segment'].unique()
# for segment_val in unique_segments:
#     plt.scatter(X_customers[customer_data['Segment'] == segment_val, 0], 
#                 X_customers[customer_data['Segment'] == segment_val, 1], 
#                 label=f'Segment {segment_val}')
# centers_cs = scaler.inverse_transform(final_kmeans_model.cluster_centers_) # Transform centers back
# plt.scatter(centers_cs[:,0], centers_cs[:,1], marker='X', s=200, color='black', label='Centroids')
# plt.title('Customer Segments')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.grid(True)
# plt.show()
```

## Summary

Clustering is a powerful unsupervised learning technique for discovering hidden patterns and structures in data. Key algorithms include:
- **K-Means**: Simple, efficient for spherical clusters. Requires `k`.
- **Hierarchical Clustering**: Creates a tree of clusters, good for understanding relationships. Can be slow for large data.
- **DBSCAN**: Finds arbitrarily shaped clusters and handles noise. Sensitive to `eps` and `min_samples`.
- **Gaussian Mixture Models (GMM)**: Probabilistic, flexible for various cluster shapes.

Choosing the right algorithm and parameters, along with proper data preprocessing and evaluation, is crucial for obtaining meaningful clusters.

## Next Chapter Preview

In Chapter 7, we'll explore neural networks, the foundation of deep learning and modern AI systems, moving from unsupervised learning to complex supervised learning models.

## Additional Resources

- [Scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html) - Official documentation with examples.
- [Cluster Analysis: Basic Concepts and Algorithms](https://www-users.cse.umn.edu/~kumar/dmbook/ch8.pdf) - Chapter from "Introduction to Data Mining" by Tan, Steinbach, Kumar.
- [A Tutorial on Clustering Algorithms by M. E. Celebi](https://www.researchgate.net/publication/262628085_A_Tutorial_on_Clustering_Algorithms)

---
**Note**: The Python code snippets are illustrative. You'll need to adapt them with actual datasets and potentially more detailed preprocessing steps for real-world applications. Variables like `X`, `true_labels`, etc., are assumed to be defined in a practical context. 