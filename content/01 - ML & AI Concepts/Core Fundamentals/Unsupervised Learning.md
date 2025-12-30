## Overview

Unsupervised learning discovers patterns in data without labeled examples. Unlike [[Supervised Learning]], there's no "correct answer" to learn from. The algorithm must find structure on its own. The core challenge is defining what "good structure" means without ground truth.

## Key Idea

**Data is not uniformly distributed**. Real-world data clusters, has underlying dimensions, and follows patterns. Unsupervised learning exploits this non-uniformity.

## Types of Unsupervised Learning

### 1. Clustering
Group data points such that points within a cluster are more similar to each other than to those in other clusters.

### 2. Dimensionality Reduction
Find a lower-dimensional representation that preserves important structure (variance, distances, neighborhoods).

### 3. Anomaly Detection
Identify data points that don't fit the learned pattern of "normal."

### 4. Association Rule Learning
Discover relationships between variables.

---

## Mathematical Foundation

### The Unsupervised Setup

We have:
- **Input space** $\mathcal{X}$: the domain of possible inputs
- **Unlabeled dataset** $D = \{x_1, x_2, ..., x_n\}$ where $x_i \in \mathcal{X}$

No labels $y_i$. The goal varies by task:
- **Clustering**: Learn a mapping $c: \mathcal{X} \rightarrow \{1, 2, ..., K\}$
- **Dimensionality Reduction**: Learn a mapping $f: \mathcal{X} \rightarrow \mathcal{Z}$ where $\dim(\mathcal{Z}) < \dim(\mathcal{X})$
- **Density Estimation**: Learn the probability distribution $P(X)$

---

## Clustering Algorithms

### K-Means Clustering

The most widely used clustering algorithm. Partitions data into $K$ clusters by minimizing within-cluster variance.

**Objective Function (Inertia)**:
$$J = \sum_{k=1}^{K}\sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

where $C_k$ is the set of points in cluster $k$ and $\mu_k$ is the centroid of cluster $k$.

**Algorithm**:
1. Initialize $K$ centroids randomly
2. **Assignment Step**: Assign each point to nearest centroid
   $$c_i = \arg\min_k \|x_i - \mu_k\|^2$$
3. **Update Step**: Recompute centroids
   $$\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$$
4. Repeat until convergence

**Why it works**: Each step monotonically decreases $J$. Assignment minimizes $J$ for fixed centroids; update minimizes $J$ for fixed assignments.

**Limitations**:
- Must specify $K$ in advance
- Sensitive to initialization (use K-Means++ for better initialization)
- Assumes spherical, equally-sized clusters
- Converges to local minimum

### Hierarchical Clustering

Builds a tree (dendrogram) of clusters. Two approaches:

**Agglomerative (Bottom-Up)**:
1. Start with each point as its own cluster
2. Repeatedly merge the two closest clusters
3. Stop when desired number of clusters reached

**Divisive (Top-Down)**:
1. Start with all points in one cluster
2. Recursively split clusters
3. Stop when desired granularity reached

**Linkage Methods** (how to measure cluster distance):
- **Single Linkage**: $d(A, B) = \min_{a \in A, b \in B} d(a, b)$ () 
- **Complete Linkage**: $d(A, B) = \max_{a \in A, b \in B} d(a, b)$ 
- **Average Linkage**: $d(A, B) = \frac{1}{|A||B|}\sum_{a \in A}\sum_{b \in B} d(a, b)$
- **Ward's Method**: Minimize increase in total within-cluster variance

### DBSCAN (Density-Based Spatial Clustering)

Clusters are dense regions separated by sparse regions. Handles arbitrary cluster shapes.

**Key Parameters**:
- $\epsilon$ (eps): Neighborhood radius
- MinPts: Minimum points to form a dense region

**Point Classification**:
- **Core Point**: Has $\geq$ MinPts within $\epsilon$ radius
- **Border Point**: Within $\epsilon$ of a core point but not itself core
- **Noise Point**: Neither core nor border

**Algorithm**:
1. Find all core points
2. Connect core points that are within $\epsilon$ of each other
3. Assign border points to nearby clusters
4. Label remaining points as noise

**Advantages**: No need to specify $K$, finds arbitrary shapes, robust to outliers.

**Limitations**: Struggles with varying density clusters, sensitive to $\epsilon$ and MinPts.

---

## Dimensionality Reduction

### [[PCA (Principal Component Analysis)]]

Find orthogonal directions (principal components) that maximize variance in the data.

**Mathematical Formulation**:
Given centered data matrix $X \in \mathbb{R}^{n \times d}$, find projection directions.

The first principal component $w_1$ maximizes:
$$w_1 = \arg\max_{\|w\|=1} \text{Var}(Xw) = \arg\max_{\|w\|=1} w^T S w$$

where $S = \frac{1}{n}X^TX$ is the covariance matrix.

**Solution**: The principal components are the eigenvectors of $S$, ordered by eigenvalue magnitude.

$$S = V \Lambda V^T$$

where $V$ contains eigenvectors as columns and $\Lambda$ is diagonal with eigenvalues $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$.

**Variance Explained**: The proportion of variance captured by the first $k$ components:
$$\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{d}\lambda_i}$$

**Projection**: To reduce to $k$ dimensions:
$$Z = X V_k$$

where $V_k$ contains the top $k$ eigenvectors.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction optimized for visualization. Preserves local neighborhood structure.

**Intuition**: Points that are close in high-dimensional space should be close in low-dimensional space.

**High-dimensional similarities** (Gaussian kernel):
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

Symmetrize: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

**Low-dimensional similarities** (t-distribution with 1 df):
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

The t-distribution has heavier tails than Gaussian, allowing moderate distances in high-d to become larger in low-d (alleviates crowding problem).

**Objective**: Minimize KL divergence between $P$ and $Q$:
$$\text{KL}(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Key Parameter**: Perplexity (roughly, effective number of neighbors to consider). Typical range: 5-50.

**Limitations**:
- Non-parametric (can't project new points directly)
- Computationally expensive: $O(n^2)$
- Results depend on random initialization
- Cluster sizes in visualization don't reflect true cluster sizes

### UMAP (Uniform Manifold Approximation and Projection)

Modern alternative to t-SNE. Based on Riemannian geometry and algebraic topology. Generally faster and better preserves global structure.

**Key Differences from t-SNE**:
- Constructs a weighted graph based on local distances
- Optimizes cross-entropy rather than KL divergence
- Can embed new points after training

---

## Anomaly Detection

### Statistical Methods

**Z-Score**: Flag points where $|z| = \frac{|x - \mu|}{\sigma} > \text{threshold}$

**Mahalanobis Distance**: Accounts for correlations
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

### Isolation Forest

Based on the principle that anomalies are "few and different".

**Algorithm**:
1. Build trees by randomly selecting features and split values
2. Anomalies require fewer splits to isolate
3. Anomaly score based on average path length across trees

**Path Length Interpretation**:
- Short path length → likely anomaly
- Long path length → likely normal

### One-Class SVM

Learn a boundary around "normal" data. Uses [[Support Vector Machines]] principles with only one class.

---

## Association Rule Learning

Discover rules like $\{bread, butter\} \Rightarrow \{milk\}$.

**Key Metrics**:
- **Support**: $\text{supp}(X) = \frac{|\{t : X \subseteq t\}|}{|D|}$  ( how frequently itemset appears )
- **Confidence**: $\text{conf}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}$ ( reliability of rule )
- **Lift**: $\text{lift}(X \Rightarrow Y) = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)}$  ( if $> 1$, positive correlation )

**Apriori Algorithm**: Prune itemsets with support below threshold, build rules from frequent itemsets.

---

## Practical Application

### When to Use Each Algorithm

| Scenario | Recommended Approach |
|----------|---------------------|
| Unknown number of clusters | Hierarchical, DBSCAN |
| Large dataset, known $K$ | K-Means, Mini-Batch K-Means |
| Arbitrary cluster shapes | DBSCAN, Spectral Clustering |
| Visualization (2D/3D) | t-SNE, UMAP |
| Feature extraction / compression | PCA |
| Noise/outlier detection | DBSCAN, Isolation Forest |
| Market basket analysis | Apriori, FP-Growth |

### Choosing Number of Clusters (K)

- **Elbow Method**: Plot inertia vs $K$, look for "elbow"
- **Silhouette Score**: Measures how similar points are to own cluster vs others. Range $[-1, 1]$, higher is better
- **Gap Statistic**: Compare within-cluster dispersion to null reference
- **Domain Knowledge**: Often the most reliable

### Common Pitfalls

- **Not scaling features**: K-Means and PCA are sensitive to feature scales. Standardize first.
- **Ignoring cluster validation**: Always validate with multiple metrics and visual inspection.
- **Over-interpreting t-SNE**: Distances between clusters are not meaningful; cluster sizes are misleading.
- **Using PCA for non-linear data**: Consider kernel PCA or t-SNE/UMAP instead.
- **Wrong distance metric**: Euclidean isn't always appropriate (e.g., text data → cosine similarity).

---

## Comparison Table

| Algorithm | Cluster Shape | Scalability | Needs K? | Handles Noise |
|-----------|---------------|-------------|----------|---------------|
| K-Means | Spherical | Very Good | Yes | No |
| Hierarchical | Any | Poor ($O(n^2)$ or worse) | No | No |
| DBSCAN | Arbitrary | Good | No | Yes |
| Gaussian Mixture | Elliptical | Good | Yes | No |
| Spectral | Arbitrary | Poor | Yes | No |

| Dim. Reduction | Linear? | Preserves | Speed | New Points |
|----------------|---------|-----------|-------|------------|
| PCA | Yes | Global variance | Fast | Yes |
| t-SNE | No | Local structure | Slow | No |
| UMAP | No | Local + Global | Medium | Yes |
| Autoencoders | No | Learned features | Varies | Yes |

---

## Related Concepts

- [[Supervised Learning]]
- [[PCA (Principal Component Analysis)]]
- [[Dimensionality Reduction]]
- [[Support Vector Machines]]
- [[Feature Engineering Principles]]

---

## Resources

### Papers
- [A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189) 
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) 
- [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426)

### Others
- [Clustering with Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html) 
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/) 
- [StatQuest: K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [StatQuest: t-SNE](https://www.youtube.com/watch?v=NEaUSP4YerM)

---

**Back to**: [[01 - Core Fundamentals Index]]
