Regression metrics are quantitative measures used to evaluate how well a machine learning model predicts continuous numerical values. 
## Core Purpose
These metrics answer the fundamental question: "How far off are the predictions from reality?" Different metrics emphasize different aspects of prediction errors.

---
## 1. Mean Absolute Error (MAE)

**Formula:** `MAE = (1/n) × Σ|yᵢ - ŷᵢ|`

**Description:** Average of absolute differences between predictions and actual values.

**Key Properties:**
- Same units as the target variable
- Treats all errors equally (linear penalty)
- Robust to outliers compared to MSE
- Easy to interpret

**When to Use:** Straightforward interpretability and all errors should be weighted equally.

**Range:** [0, ∞) where 0 is perfect prediction

---

## 2. Mean Squared Error (MSE)

**Formula:** `MSE = (1/n) × Σ(yᵢ - ŷᵢ)²`

**Description:** Average of squared differences between predictions and actual values.

**Key Properties:**
- Units are squared (less intuitive)
- Penalizes larger errors more heavily (quadratic penalty)
- Sensitive to outliers
- Differentiable everywhere (good for optimization)

**When to Use:** When large errors are particularly undesirable.

**Range:** [0, ∞) where 0 is perfect prediction

---

## 3. Root Mean Squared Error (RMSE)

**Formula:** `RMSE = √MSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]`

**Description:** Square root of MSE, bringing error back to original units.

**Key Properties:**
- Same units as the target variable
- Maintains sensitivity to large errors like MSE
- More interpretable than MSE
- Most widely used regression metric

**When to Use:** Default choice for most regression problems; balances interpretability with sensitivity to outliers.

**Range:** [0, ∞) where 0 is perfect prediction

---

## 4. R-squared (R² / Coefficient of Determination)

**Formula:** `R² = 1 - (SS_res / SS_tot)` where `SS_res = Σ(yᵢ - ŷᵢ)²` and `SS_tot = Σ(yᵢ - ȳ)²`

**Description:** Proportion of variance in the dependent variable explained by the model.

**Key Properties:**
- Scale-independent
- Ranges from -∞ to 1 (typically 0 to 1 for reasonable models)
- 1 = perfect prediction, 0 = model performs no better than mean
- Can be negative if model performs worse than predicting the mean
- Doesn't tell you if your errors are large or small in absolute terms - it tells you if your errors are small _relative to the natural variation in the data_.

**When to Use:** When you want to understand overall explanatory power of the model.

**Range:** (-∞, 1] where 1 is perfect prediction

---

## 5. Adjusted R-squared

**Formula:** `Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]`

where `n` = number of samples, `p` = number of predictors

**Description:** Modified R² that penalizes addition of unhelpful features.

**Key Properties:**
- Accounts for number of predictors in the model
- Always lower than or equal to R²
- Better for comparing models with different numbers of features
- Can be negative

**When to Use:** Comparing models with different numbers of features or avoiding overfitting.

**Range:** (-∞, 1] where 1 is perfect prediction

---

## 6. Mean Absolute Percentage Error (MAPE)

**Formula:** `MAPE = (100/n) × Σ|((yᵢ - ŷᵢ) / yᵢ)|`

**Description:** Average of absolute percentage errors.

**Key Properties:**
- Scale-independent (expressed as percentage)
- Easy to interpret and communicate
- Cannot be used when actual values are zero
- Asymmetric (penalizes over-predictions more than under-predictions)

**When to Use:** Need scale-independent comparison across datasets or percentage error is meaningful.

**Range:** [0, ∞) where 0 is perfect prediction

---

## 7. Mean Squared Logarithmic Error (MSLE)

**Formula:** `MSLE = (1/n) × Σ(log(1 + yᵢ) - log(1 + ŷᵢ))²`

**Description:** MSE applied to logarithm of predictions and actual values.

**Key Properties:**
- Penalizes under-predictions more than over-predictions
- Useful when target spans several orders of magnitude
- Cares about relative rather than absolute differences
- Only works with non-negative values

**When to Use:** To Predict values across wide ranges or when relative error matters more than absolute error.

**Range:** [0, ∞) where 0 is perfect prediction


---

## 8. Median Absolute Error (MedAE)

**Formula:** `MedAE = median(|yᵢ - ŷᵢ|)`

**Description:** Median of absolute differences between predictions and actual values.

**Key Properties:**
- Same units as target variable
- Highly robust to outliers
- Not differentiable (less useful for optimization)
- Better represents "typical" error

**When to Use:** When your data has significant outliers that shouldn't dominate the metric.

**Range:** [0, ∞) where 0 is perfect prediction

---

## 9. Huber Loss

**Formula:** 
```
L(y, ŷ) = {
  0.5 × (y - ŷ)²           for |y - ŷ| ≤ δ
  δ × (|y - ŷ| - 0.5δ)     otherwise
}
```

**Description:** Quadratic for small errors, linear for large errors.

**Key Properties:**
- Less sensitive to outliers than MSE
- Differentiable everywhere (good for gradient-based optimization)
- Requires tuning of δ parameter
- Combines benefits of MSE and MAE

**When to Use:** When you want MSE-like behavior for small errors but robustness to outliers.

**Range:** [0, ∞) where 0 is perfect prediction

---

## 10. Max Error

**Formula:** `Max Error = max(|yᵢ - ŷᵢ|)`

**Description:** Maximum absolute error across all predictions.

**Key Properties:**
- Shows worst-case prediction
- Extremely sensitive to outliers
- Same units as target variable
- Useful for safety-critical applications

**When to Use:** To ensure no single prediction exceeds a threshold.

**Range:** [0, ∞) where 0 is perfect prediction

---

## Quick Comparison Table

| Metric      | Units       | Outlier Sensitivity | Interpretability | Range   |
| ----------- | ----------- | ------------------- | ---------------- | ------- |
| MAE         | Original    | Low                 | High             | [0, ∞)  |
| MSE         | Squared     | High                | Medium           | [0, ∞)  |
| RMSE        | Original    | High                | High             | [0, ∞)  |
| R²          | None        | Medium              | High             | (-∞, 1] |
| Adjusted R² | None        | Medium              | High             | (-∞, 1] |
| MAPE        | Percentage  | Medium              | High             | [0, ∞)  |
| MSLE        | Squared Log | Medium              | Medium           | [0, ∞)  |
| MedAE       | Original    | Very Low            | High             | [0, ∞)  |
| Huber       | Original    | Low-Medium          | Medium           | [0, ∞)  |
| Max Error   | Original    | Very High           | High             | [0, ∞)  |

---
**Back to**: [[ML & AI Index]]
