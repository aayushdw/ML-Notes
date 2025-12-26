## Overview
Feature Engineering is mathematical transformation of the input space $\mathcal{X}$ to a latent space $\mathcal{X}'$ that maximizes the predictive power of a hypothesis function $h_\theta(x)$ 

## Mathematical Transformations (Numerical)

### Scaling and Magnitude
Numerical data is the native language of most algorithms (especially Neural Networks). The goal is to optimize the **loss landscape** for faster convergence and to satisfy statistical assumptions (e.g., normality)

Many algorithms (SVMs, k-NN, Linear Regression with Regularization, Neural Networks) are sensitive to the scale of input features. If feature $x_1$ ranges from $[0, 1]$ and $x_2$ ranges from $[0, 1000]$, the cost function contours become elongated ellipses rather than circles, making Gradient Descent oscillate and converge slowly.
#### Min-Max Normalization
Rescales the data to a fixed range, usually $[0, 1]$.

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

- **Use case:** When you need bounded intervals (e.g., for Image pixel intensities) or algorithms that do not assume any distribution (k-NN).
- **Drawback:** Highly sensitive to outliers. A single large outlier shifts the majority of data into a tiny interval.

#### Standardization (Z-Score)
Rescales data to have a mean ($\mu$) of 0 and standard deviation ($\sigma$) of 1.

$$z = \frac{x - \mu}{\sigma}$$

- **Use case:** The default for most linear models and Neural Networks. It preserves outliers but centers the data, allowing weights to learn symmetrically around zero.

### Transformations for Distribution (Gaussianity)
Linear models and Neural Networks often perform better when features are normally distributed (Gaussian). Skewed data (long tails) can result in models focusing too much on the "head" of the distribution and ignoring the "tail" (rare high-value events).
#### Log Transformation
Compresses the range of large numbers.

$$x' = \ln(1 + x)$$

- **Note:** We add $1$ so that if $x=0$, $\ln(1)=0$. This is crucial for "count" data (e.g., number of clicks).
#### Power Transforms (Box-Cox & Yeo-Johnson)
These are parametric transformations that attempt to make data resemble a normal distribution by finding an optimal lambda parameter $\lambda$.

The **Box-Cox** transformation requires $x > 0$:

$$x^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(x) & \text{if } \lambda = 0 \end{cases}$$


The **Yeo-Johnson** transformation supports positive and negative values:

$$x^{(\lambda)} = \begin{cases} \frac{(x+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \geq 0 \\ \ln(x+1) & \text{if } \lambda = 0, x \geq 0 \\ \dots & (\text{handled for } x < 0) \end{cases}$$

### Handling Outliers & Nonlinearity

#### Winsorization (Clipping)
Capping feature values at specific percentiles (e.g., 1st and 99th).

$$x' = \max(P_{01}, \min(x, P_{99}))$$

#### Discretization (Binning)
Converting continuous variables into discrete buckets. This introduces non-linearity into linear models without increasing model complexity significantly.
- **Equal-width:** Divides the range into $N$ intervals of the same size.
- **Equal-frequency (Quantile):** Divides data so each bin has the same number of samples. (Generally preferred as it handles outliers better).

## Categorical Encoding
Mathematical models operate on vector spaces $\mathbb{R}^n$. Categorical data (strings, labels) exists in a symbolic space $\mathcal{S}$. Encoding is the mapping function $f: \mathcal{S} \rightarrow \mathbb{R}^n$ that preserves the information content of the symbol while making it digestible for the hypothesis function.

The core challenge in categorical encoding is the **Bias-Variance Tradeoff** regarding **Cardinality** (the number of unique categories).

### Handling Low Cardinality
#### One-Hot Encoding (OHE)
Maps a variable with cardinality $K$ to a vector of size $K$.
$$x \rightarrow [0, 0, \dots, 1, \dots, 0] \in \{0, 1\}^K$$
- **Mathematical Implication:** It assumes **orthogonality**. By placing every category on a separate axis, you are telling the model that the distance between "Apple" and "Banana" is exactly equal to the distance between "Apple" and "Car".
    
- **The "Curse of Dimensionality":** For high cardinality, OHE creates a sparse matrix where data points become equidistant, making it hard for distance-based algorithms (k-NN, SVMs) to find meaningful patterns.
    
- **Tree-based models:** While Trees can handle OHE, if $K$ is very large, the tree must grow very deep to isolate specific categories, leading to fragmentation of the dataset and overfitting.

#### Ordinal Encoding
Maps categories to integers $[1, 2, \dots, K]$ based on an inherent order (e.g., "Low", "Medium", "High").
- **Constraint:** You impose a prior belief that $Medium - Low = High - Medium$. If this linear relationship doesn't exist in reality, you introduce bias.

### Handling High Cardinality (Bayesian Encoders)
When $K$ is high (e.g., User IDs, Zip Codes), OHE fails. We turn to **Target Encoding** (also known as Mean Encoding).
The Concept: Replace the category $x_i$ with a scalar representing the likelihood of the target $Y$ given that category.

$$x'_i = P(Y=1 | x_i)$$


---
**Back to**: [[ML & AI Index]]
