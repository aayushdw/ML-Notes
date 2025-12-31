# Decision Trees

## Overview

Decision Trees are a fundamental [[Supervised Learning]] algorithm that learns a hierarchy of if-then rules to partition the feature space into regions, each assigned a prediction. They can handle both classification and regression tasks. The model structure resembles an inverted tree, where internal nodes represent feature tests, branches represent outcomes, and leaf nodes contain predictions.

The key appeal of decision trees lies in their interpretability: you can trace exactly why a prediction was made by following the path from root to leaf. This makes them invaluable when model explainability is a requirement.


## Core Mental Model

Decision Trees are like flowchart for making predictions. At each step, you ask a yes/no question about a feature, and based on the answer, you proceed down the appropriate branch until you reach a final decision.

**Example**: Predicting whether someone will buy a product:

![[Decision Trees 2025-12-30 16.52.26.excalidraw.svg]]

### Recursive Binary Splitting

The algorithm builds the tree top-down through **recursive partitioning**:

1. Start with all training data at the root
2. Find the best feature and threshold to split the data (minimizes impurity)
3. Partition data into left and right child nodes
4. Recursively repeat steps 2-3 for each child
5. Stop when a stopping criteria is met (max depth, min samples, pure node)

### Feature Space Partitioning

Decision trees create axis-aligned decision boundaries. Each split divides the feature space with a line perpendicular to one axis.

![[Decision Trees 2025-12-30 16.58.27.excalidraw.svg]]

This axis-aligned nature is both a strength (easy to interpret) and a limitation (cannot capture diagonal boundaries efficiently).

---

## Mathematical Foundation

### Measuring Impurity

The core question at each node is: **"Which split best separates the classes?"** We need a metric to quantify how "mixed" or "impure" a node is.

Let $p_k$ denote the proportion of class $k$ samples at a node. For a binary classification problem, let $p$ be the proportion of the positive class.

#### Gini Impurity

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

For binary classification:

$$\text{Gini}(t) = 1 - p^2 - (1-p)^2 = 2p(1-p)$$

**Interpretation**: Gini impurity measures the probability of misclassifying a randomly chosen sample if it were labeled randomly according to the class distribution at that node.

- **Gini = 0**: Pure node (all samples belong to one class)
- **Gini = 0.5** (binary): Maximum impurity (50-50 split)

#### Entropy (Information Gain)

$$\text{Entropy}(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

For binary classification:

$$\text{Entropy}(t) = -p \log_2(p) - (1-p) \log_2(1-p)$$

**Interpretation**: Entropy measures the uncertainty or disorder in the class distribution. It originates from information theory, representing the expected number of bits needed to encode class membership.

- **Entropy = 0**: Pure node
- **Entropy = 1** (binary): Maximum uncertainty

### Information Gain

When evaluating a split on feature $f$ at threshold $\tau$, we compute the **reduction in impurity**:

$$\text{Gain}(t, f, \tau) = I(t) - \frac{|t_L|}{|t|} I(t_L) - \frac{|t_R|}{|t|} I(t_R)$$

where:
- $I$ is the impurity function (Gini or Entropy)
- $t$ is the parent node
- $t_L$, $t_R$ are left and right child nodes
- $|t|$ denotes the number of samples at node $t$

The algorithm greedily selects the split $(f^*, \tau^*)$ that maximizes this gain:

$$(f^*, \tau^*) = \arg\max_{f, \tau} \text{Gain}(t, f, \tau)$$

### Gini vs Entropy: Does it Matter?

In practice, both criteria yield very similar trees. Gini is slightly faster to compute (no logarithm). The choice rarely affects model performance significantly.

![[Decision Trees 2025-12-30 17.11.19.excalidraw.svg]]


### Regression Trees: Variance Reduction

For regression tasks, we cannot use class proportions. Instead, we minimize the **mean squared error** within each node:

$$\text{MSE}(t) = \frac{1}{|t|} \sum_{i \in t} (y_i - \bar{y}_t)^2$$

where $\bar{y}_t$ is the mean target value at node $t$.

The prediction for a leaf node is simply the mean of the target values: $\hat{y} = \bar{y}_t$.

The split criterion becomes **variance reduction**:

$$\text{Gain}(t, f, \tau) = \text{MSE}(t) - \frac{|t_L|}{|t|} \text{MSE}(t_L) - \frac{|t_R|}{|t|} \text{MSE}(t_R)$$

---

## Algorithm Details

### CART (Classification and Regression Trees)

The most common algorithm, used by scikit-learn:

```
function BuildTree(D, depth):
    if stopping_condition(D, depth):
        return LeafNode(majority_class(D) or mean(D))

    best_gain = 0
    for each feature f in features:
        for each threshold τ in unique_values(D[f]):
            gain = compute_gain(D, f, τ)
            if gain > best_gain:
                best_gain = gain
                best_f, best_τ = f, τ

    D_left = {x ∈ D : x[best_f] <= best_τ}
    D_right = {x ∈ D : x[best_f] > best_τ}

    return InternalNode(
        feature = best_f,
        threshold = best_τ,
        left = BuildTree(D_left, depth + 1),
        right = BuildTree(D_right, depth + 1)
    )
```

### Handling Categorical Features

Two approaches:
1. **One-hot encoding**: Convert to binary features (standard approach)
2. **Native categorical splits**: Some implementations (LightGBM) support direct categorical splits

### Missing Values

Different strategies:
- **Surrogate splits**: Find alternative features that produce similar splits
- **Separate branch**: Create a third branch for missing values
- **Imputation**: Fill missing values before training

---

## Practical Application

### When to Use Decision Trees

- **Interpretability is critical**: Medical diagnosis, loan approval, legal decisions
- **Feature interactions matter**: Trees naturally capture interactions without explicit feature engineering
- **Mixed feature types**: Handles numerical and categorical features naturally
- **Quick baseline**: Fast to train, provides a reasonable starting point
- **Non-linear relationships**: Can model complex non-linear patterns
- **Exploratory analysis**: Reveals important features and their thresholds

### When NOT to Use

- **Smooth decision boundaries needed**: Trees create jagged, axis-aligned boundaries
- **High-dimensional sparse data**: Struggle with many irrelevant features (consider [[Random Forests]])
- **Extrapolation required**: Trees cannot predict outside the range of training data
- **Stability is important**: Small data changes can produce very different trees
- **Maximum accuracy needed**: Ensemble methods like [[Gradient Boosting]] or [[Random Forests]] almost always outperform single trees

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Overfitting** | Deep trees memorize training data | Limit depth, prune, use min_samples_leaf |
| **High variance** | Small data changes yield different trees | Use [[Ensemble Methods]] |
| **Feature scale sensitivity** | Actually, trees are invariant to feature scaling | Not a problem for trees (unlike [[Support Vector Machines]]) |
| **Imbalanced classes** | Trees bias toward majority class | Use class_weight, balanced sampling |
| **Ignoring feature importance** | Missing insights from the model | Always check feature_importances_ |

### Key Hyperparameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `max_depth` | Controls tree complexity | 3-20 (None for unlimited) |
| `min_samples_split` | Minimum samples to split a node | 2-20 |
| `min_samples_leaf` | Minimum samples in a leaf | 1-10 |
| `max_features` | Features considered per split | sqrt, log2, or fraction |
| `criterion` | Impurity measure | gini, entropy (classification) |
| `ccp_alpha` | Cost-complexity pruning parameter | 0.0-0.1 |

### Pruning

Pruning prevents overfitting by removing branches that provide little predictive power:

#### Pre-pruning (early stopping):
- Set max_depth, min_samples_split, min_samples_leaf before training

#### Post-pruning (cost-complexity pruning):
- Grow full tree, then prune branches that minimally increase training error
- Controlled by `ccp_alpha`: higher values = more pruning

The cost-complexity criterion for subtree $T_t$ rooted at node $t$:

$$R_\alpha(T) = R(T) + \alpha |T|$$

where $R(T)$ is the tree's error and $|T|$ is the number of leaves. We prune if:
$$R_\alpha(t) \leq R_\alpha(T_t)$$

![[Decision Trees 2025-12-30 17.35.59.excalidraw.svg]]

### Computational Complexity

- **Training**: $O(n \cdot m \cdot \log n)$ where $n$ = samples, $m$ = features
- **Prediction**: $O(\log n)$ per sample (tree depth)
- **Memory**: $O(\text{nodes})$, typically much smaller than dataset

## Feature Importance

Decision trees provide built-in feature importance scores based on the total impurity decrease:

$$\text{Importance}(f) = \sum_{t: \text{split on } f} \frac{|t|}{|T|} \cdot \text{Gain}(t)$$

where the sum is over all nodes that split on feature $f$, and $|T|$ is the total number of training samples.

**Caveat**: This can be biased toward high-cardinality features. Consider permutation importance for more reliable estimates.

## Comparisons

| Aspect | Decision Tree | [[Random Forests]] | [[Gradient Boosting]] |
|--------|--------------|-------------------|----------------------|
| **Bias** | Low | Low | Low |
| **Variance** | High | Low | Low |
| **Interpretability** | High | Medium | Low |
| **Training Speed** | Fast | Medium | Slow |
| **Overfitting Risk** | High | Low | Medium |
| **Hyperparameter Sensitivity** | Medium | Low | High |
| **Handles Missing Values** | Some implementations | Yes (random) | Yes |

### Decision Tree vs Logistic Regression

| Criteria | Decision Tree | Logistic Regression |
|----------|--------------|---------------------|
| Decision boundary | Non-linear, axis-aligned | Linear |
| Interpretability | Rule-based | Coefficient-based |
| Feature interactions | Automatic | Must specify manually |
| Probability calibration | Often poor | Generally good |
| Overfitting tendency | High | Low |

---
## Related Concepts

- [[Random Forests]]: Ensemble of decision trees with bagging
- [[Gradient Boosting]]: Sequential ensemble that corrects errors
- [[Ensemble Methods]]: General framework for combining models
- [[Overfitting and Underfitting]]: Key concern for decision trees
- [[Bias-Variance Tradeoff]]: Trees have low bias but high variance
- [[Feature Engineering Principles]]: Trees reduce need for feature engineering

---

## Resources

### Papers
- [Classification and Regression Trees (Breiman et al., 1984)](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418) 
- [C4.5: Programs for Machine Learning (Quinlan, 1993)](https://link.springer.com/book/10.1007/978-1-4899-7502-7) 
### Articles
- [scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [A Visual Introduction to Decision Trees](https://mlu-explain.github.io/decision-tree/) 

### Videos
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk) 
- [StatQuest: Regression Trees](https://www.youtube.com/watch?v=g9c66TUylZ4) 

---

**Back to**: [[01 - Core Fundamentals Index]]
