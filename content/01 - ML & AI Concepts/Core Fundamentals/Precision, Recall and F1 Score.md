## Overview
These are classification metrics used to evaluate how well a binary classification model performs. While accuracy is the most intuitive metric, it can be misleading on imbalanced datasets. Precision, Recall, and F1-Score provide a more nuanced view of the model's errors.

## Intuition
### The "Fishing Net" Analogy
Imagine you are a fisherman trying to catch a specific type of high-value fish (e.g., Tuna) in a lake that also contains junk (boots, cans) and other fish.

*   **Precision (Quality of Catch)**:
    *   *Question*: "Of all the things caught in my net, what percentage are actually Tuna?"
    *   *Goal*: Minimize **False Positives** (don't catch junk).
    *   *High Precision*: You catch very few things, but everything you catch is definitely a Tuna. You might miss many Tuna in the lake, but you don't have any trash.

*   **Recall (Quantity of Catch)**:
    *   *Question*: "Of all the Tuna in the entire lake, what percentage did I manage to catch?"
    *   *Goal*: Minimize **False Negatives** (don't let Tuna escape).
    *   *High Recall*: You use a massive net and catch every single Tuna in the lake. However, you also catch a lot of boots and cans.

### Visual Understanding
*   **Aggressive Model (High Recall)**: "Flag everything that looks remotely like a fish." (Catches all fish, but high noise).
*   **Conservative Model (High Precision)**: "Only flag it if you are 100% sure it's a fish." (Very clean results, but misses hard-to-spot fish).

## Mathematical Foundation
The foundation of these metrics is the **Confusion Matrix**.

$$
\begin{array}{c|cc}
& \text{Predicted Negative} & \text{Predicted Positive} \\
\hline
\text{Actual Negative} & \text{TN} & \text{FP (Type I Error)} \\
\text{Actual Positive} & \text{FN (Type II Error)} & \text{TP}
\end{array}
$$

### Core Equations

**1. Precision**
The fraction of positive predictions that are actually correct.

$$ \text{Precision} = \frac{TP}{TP + FP} $$

**2. Recall (Sensitivity)**
The fraction of actual positives that were identified correctly.

$$ \text{Recall} = \frac{TP}{TP + FN} $$

**3. F1-Score**
The harmonic mean of Precision and Recall. It punishes extreme values (e.g., if Precision is 100% but Recall is 0%, F1 will be 0).

$$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

**4. Accuracy**
The fraction of total predictions that were correct.

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

## Practical Application

### When to Use Which? ("The Cost of Error")

| Metric Strategy | Scenario | Example | Why? |
| :--- | :--- | :--- | :--- |
| **Maximize Recall** | **FN is expensive/dangerous** | Cancer Diagnosis, Terrorist Detection | Missing a sick patient (FN) is worse than testing a healthy one (FP). |
| **Maximize Precision** | **FP is annoying/costly** | Spam Filter, YouTube Recommendations | Blocking a legit email (FP) is worse than letting one spam email through (FN). |
| **Maximize F1-Score** | **Balance needed** | Fraud Detection, Imbalanced Data | You want to catch fraud (Recall) but not block valid users (Precision). |

### Common Pitfalls
*   **Imbalanced Data Paradox**: In a dataset where 99% of samples are Negative, a model that *always* predicts Negative has 99% Accuracy but 0% Recall and undefined Precision. **Never use Accuracy for imbalanced datasets.**
*   **The Trade-off**: Increasing Precision usually decreases Recall, and vice-versa. You usually cannot maximize both simultaneously; you must pick a threshold based on business needs.

## Comparisons

| Feature | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Focus** | Overall correctness | Correctness of Positive predictions | Coverage of Actual Positives | Balance of Precision & Recall |
| **Best for** | Balanced classes | High cost of False Positives | High cost of False Negatives | Imbalanced classes / General performance |
| **Weakness** | Fails on imbalanced data | Ignores False Negatives | Ignores False Positives | Harder to interpret intuitively |

## Resources


**Back to**: [[ML & AI Index]]
