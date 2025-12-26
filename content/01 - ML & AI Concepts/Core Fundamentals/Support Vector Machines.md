# Support Vector Machines

## Overview
Support Vector Machines find the optimal hyperplane that separates classes with the maximum margin. The "support vectors" are the data points closest to the decision boundary—they're the critical examples that define where the boundary goes.

## Key Ideas
- **Maximum Margin**: Among all possible separating hyperplanes, choose the one with the widest margin between classes
- **Support Vectors**: The critical data points that lie on the margin boundaries and define the decision boundary
- **Kernel Trick**: Implicitly maps data to higher dimensions to handle non-linear decision boundaries without explicitly computing the transformation

## Core Concept

**Key Intuition**: Among all possible lines/planes that separate the classes, choose the one that's as far as possible from the nearest points of both classes. This maximizes the "safety buffer" and typically leads to better generalization.

### Visual Understanding

Imagine two groups of points (red and blue). Many lines could separate them, but SVM finds the line with the widest "street" between the groups. The points touching the edges of this street are the support vectors.

```
     blue                          
  •     •     ║               
    •       • ║    margin        
  •     •     ║               
══════════════╬══════════════  ← decision boundary
  ○     ○     ║               
    ○       ○ ║    margin        
  ○     ○     ║               
     red
```

## Mathematical Foundation

### Linear SVM (Separable Case)

We want to find a hyperplane defined by weights $w$ and bias $b$:
$$f(x) = w^T x + b$$

**Decision rule**: Classify as positive if $f(x) \geq 0$, negative otherwise.

**The optimization problem**:
$$\min_{w,b} \frac{1}{2}\|w\|^2$$

Subject to constraints:
$$y_i(w^T x_i + b) \geq 1 \quad \text{for all } i$$

This ensures all points are correctly classified with at least margin 1.

**Margin**: The distance from the hyperplane to the nearest point is $\frac{1}{\|w\|}$. Minimizing $\|w\|^2$ maximizes this margin.

### Soft-Margin SVM (Non-Separable Case)

Real data isn't perfectly separable. We introduce **slack variables** $\xi_i$ to allow some mistakes:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to:
$$y_i(w^T x_i + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0$$

**C parameter** (regularization):
- **Large C**: Few violations allowed → narrow margin, risk overfitting
- **Small C**: More violations allowed → wide margin, risk underfitting

This is the **bias-variance tradeoff** for SVMs!

### The Kernel Trick

For non-linearly separable data, SVMs use the **kernel trick** to implicitly map data to higher dimensions where it becomes separable.

**Key insight**: We don't need to explicitly compute the high-dimensional transformation. We only need the **dot product** in that space, which the kernel computes efficiently.

#### Common Kernels

**Linear Kernel**: 
$$K(x_i, x_j) = x_i^T x_j$$
- No transformation, fastest
- Use when data is already linearly separable

**Polynomial Kernel**: 
$$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$$
- Creates polynomial feature combinations
- Degree $d$ controls complexity

**RBF (Radial Basis Function/Gaussian)**: 
$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$
- Most popular, can model any decision boundary
- $\gamma$ controls influence radius of support vectors
- **Large $\gamma$**: nearby points matter most → complex boundary, may overfit
- **Small $\gamma$**: distant points matter → smoother boundary
- Maps points into Infinite dimensional space (Taylor Series Expansion of exp(z) function )

**Sigmoid Kernel**: 
$$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$$
- Similar to neural network activation

### Why Support Vectors Matter

The solution only depends on a subset of training points (support vectors):

$$f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$$

where $\alpha_i > 0$ only for support vectors. This makes SVMs:
- **Memory efficient**: Only store support vectors
- **Robust**: Outliers far from boundary don't affect the decision
## When to Use SVMs

### Good For
- Small to medium datasets (< 10,000 samples)
- High-dimensional data (many features, few samples)
- **Clear margin of separation exists**
- **Binary classification** (natural formulation)
- When you need probabilistic outputs (with `probability=True`)

### Not Ideal For
- Very large datasets (slow to train, O(n²) to O(n³))
- Noisy data with overlapping classes
- **Multi-class problems** (requires one-vs-one or one-vs-rest)
- When interpretability is crucial (non-linear SVMs are black boxes)

## Intuition
Think of SVM as finding the "safest" decision boundary. If you're drawing a line to separate two groups of points, you want that line to be as far as possible from both groups. This creates a buffer zone (the margin) where you're confident about classifications.

The support vectors are like the "most difficult" examples - the ones closest to the boundary that, if removed, would change where the boundary is drawn. All other points could be removed without affecting the decision boundary.

## Advantages
- Effective in high dimensions - works well even when # features > # samples
- Memory efficient - only uses support vectors
- Versatile - different kernels for different data structures
- Robust to overfitting (especially in high-dimensional space with right C)
- Strong theoretical foundation - based on statistical learning theory

## Limitations
- Computationally expensive for large datasets
- **Sensitive to feature scaling** - ALWAYS scale features!
- No probabilistic interpretation by default (must enable, adds overhead)
- Hyperparameter tuning crucial - performance very sensitive to C, $\gamma$
- Black box for non-linear kernels - hard to interpret
- Multi-class classification requires additional strategies

## Common Pitfalls
- **Forgetting to scale features** - This is the #1 mistake! SVMs are very sensitive to feature scales
- **Using default parameters without tuning** - C and gamma need careful tuning
- **Not checking support vector ratio** - If >50% of data are support vectors, model may not be working well
- **Using RBF kernel on linearly separable data** - Try linear kernel first, it's much faster
- **Ignoring class imbalance** - Use `class_weight='balanced'` for imbalanced datasets

## Best Practices
- Always scale features using StandardScaler or MinMaxScaler
- Start with RBF kernel - most versatile for unknown data structure
- Use GridSearchCV for hyperparameter tuning - C and gamma interact
- Try linear kernel first - if it works, it's much faster
- Check class imbalance - use `class_weight='balanced'` if needed
- Monitor support vector count - too many suggests poor fit
- Use cross-validation for reliable performance estimates

## Comparison with Other Algorithms

| Aspect               | SVM            | Logistic Regression | Random Forest |
| -------------------- | -------------- | ------------------- | ------------- |
| **Speed**            | Slow (large n) | Fast                | Medium        |
| **Interpretability** | Low (kernel)   | High                | Medium        |
| **High-dim data**    | Excellent      | Good                | Good          |
| **Large datasets**   | Poor           | Excellent           | Good          |
| **Feature scaling**  | Required       | Recommended         | Not needed    |
| **Probabilistic**    | Optional       | Native              | Native        |
| **Overfitting risk** | Medium         | Low                 | Low-Medium    |

## Related Concepts
- [[Supervised Learning]]

## Resources
- https://www.youtube.com/watch?v=tV5X77lGYP4 - RBF Kernel

---
**Back to**: [[ML & AI Index]]
