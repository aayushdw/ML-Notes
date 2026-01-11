## Overview
Model trained using labeled data.

## Types
- Classification: Predicting Discrete Categories
- Regression: Predicting Continuous Values

## Mathematical Foundation

### Formal Setup
- **Input space** $\mathcal{X}$: the domain of possible inputs
- **Output space** $\mathcal{Y}$: the domain of possible outputs/labels
- **Training dataset** $D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$ where $x_i \in \mathcal{X}$ and $y_i \in \mathcal{Y}$ 
- **Hypothesis function** $h: \mathcal{X} \rightarrow \mathcal{Y}$ (our model) 

The goal is to find a hypothesis $h$ from some hypothesis space $\mathcal{H}$ that generalizes well to unseen data.
### The Learning Objective
We assume there exists some true (but unknown) function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that generates our labels, possibly with noise. Our training data comes from some joint distribution $P(X, Y)$.
**Goal**: Find $h^* \in \mathcal{H}$ that minimizes the **expected risk** (generalization error): 

$$R(h) = \mathbb{E}_{(x,y) \sim P(X,Y)}[\mathcal{L}(h(x), y)]$$
where $\mathcal{L}$ is a loss function measuring prediction error. Since we don't know $P(X,Y)$, we minimize the **empirical risk** instead: 

$$\hat{R}(h) = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(h(x_i), y_i)$$

### Common Loss Functions
**For Regression** ($\mathcal{Y} = \mathbb{R}$):
- **Mean Squared Error (MSE)**: $\mathcal{L}(h(x), y) = (h(x) - y)^2$
- **Mean Absolute Error**: $\mathcal{L}(h(x), y) = |h(x) - y|$ 
 
 **For Binary Classification** ($\mathcal{Y} = \{0, 1\}$):
 - **0-1 Loss**: $\mathcal{L}(h(x), y) = \mathbb{1}[h(x) \neq y]$ (not differentiable) 
 - **Cross-Entropy Loss**: $\mathcal{L}(\hat{y}, y) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ where $\hat{y} = h(x) \in [0,1]$ represents predicted probability. 
 
 **For Multi-class Classification** ($\mathcal{Y} = \{1, 2, ..., K\}$): 
 - **Categorical Cross-Entropy**: $\mathcal{L}(\hat{y}, y) = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$ where $y$ is one-hot encoded and $\hat{y}$ is a probability distribution over classes.

### [[Gradient Descent and Optimization]]
To minimize empirical risk, we use gradient descent. For a parameterized model $h_\theta(x)$ with parameters $\theta$: 

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \hat{R}(\theta)$$

where $\eta$ is the learning rate.

If $\theta = [\theta_0, \theta_1, ..., \theta_p]$, then: 

$$\nabla_\theta \hat{R}(\theta) = \begin{bmatrix} \frac{\partial \hat{R}}{\partial \theta_0} \\ \frac{\partial \hat{R}}{\partial \theta_1} \\ \vdots \\ \frac{\partial \hat{R}}{\partial \theta_p} \end{bmatrix}$$


Each component shows how much the loss changes when adjusting that specific parameter.

$$\nabla_\theta \hat{R}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\nabla_\theta \mathcal{L}(h_\theta(x_i), y_i)$$

[[SGD (Stochastic Gradient Descent)]] approximates this using mini-batches: 

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \frac{1}{|B|}\sum_{i \in B}\nabla_\theta \mathcal{L}(h_\theta(x_i), y_i)$$

where $B$ is a randomly sampled batch.

### Example: Linear Regression
Hypothesis: $h_\theta(x) = \theta^T x = \sum_{j=0}^{d}\theta_j x_j$ (where $x_0 = 1$ for bias)
Loss: MSE: $\mathcal{L}(h_\theta(x), y) = (h_\theta(x) - y)^2$
Gradient for one example: 

$$\nabla_\theta \mathcal{L} = 2(h_\theta(x) - y)x$$
Update rule: 

$$\theta := \theta - \eta \cdot 2(h_\theta(x) - y)x$$
**Closed-form solution** exists for linear regression: 

$$\theta^* = (X^T X)^{-1}X^T y$$

where $X$ is the design matrix and $y$ is the vector of labels.

#### [[Bias-Variance Tradeoff]]
The expected error for MSE loss function decomposes as: 
$$\mathbb{E}[(h(x) - y)^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias**: Error from wrong assumptions (under-fitting)
- **Variance**: Error from sensitivity to training data fluctuations (overfitting)
- **Irreducible Error**: Noise in the data itself 

[[Regularization Techniques]] help balance this tradeoff by adding a penalty term: 

$$\hat{R}_{reg}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(h_\theta(x_i), y_i) + \lambda \Omega(\theta)$$
Common choices: $\Omega(\theta) = \|\theta\|_2^2$ (L2/Ridge) or $\|\theta\|_1$ (L1/Lasso).



## Comparison Table

| Algorithm                       | Interpretability | Speed     | Accuracy  | Overfitting Risk | Best For                   |
| ------------------------------- | ---------------- | --------- | --------- | ---------------- | -------------------------- |
| **Logistic Regression**         | High             | Fast      | Medium    | Low              | Baseline, linear problems  |
| **Decision Trees**              | High             | Fast      | Medium    | High             | Interpretability           |
| **Random Forest**               | Medium           | Medium    | High      | Low              | General purpose            |
| **Gradient Boosting**           | Low              | Slow      | Very High | Medium           | Competitions, tabular data |
| **[[Support Vector Machines]]** | Low              | Slow      | High      | Medium           | High-dimensional           |
| **Naive Bayes**                 | High             | Very Fast | Medium    | Low              | Text, categorical data     |
| **KNN**                         | High             | Slow      | Medium    | High             | Small datasets             |
| **Neural Networks**             | Very Low         | Slow      | Very High | High             | Complex patterns, images   |

## Related Concepts
- [[Support Vector Machines]]

---
**Back to**: [[ML & AI Index]]
