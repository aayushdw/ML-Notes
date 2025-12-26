## What is ROC?
**ROC (Receiver Operating Characteristic)** visualizes binary classifier performance across ALL possible thresholds by plotting:
- **Y-axis**: TPR (True Positive Rate) = TP / (TP + FN) — also called Sensitivity/Recall
- **X-axis**: FPR (False Positive Rate) = FP / (FP + TN)

## What is AUC?
**AUC (Area Under the ROC Curve)** is a single number (0 to 1) summarizing model quality:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Very good
- **AUC = 0.7-0.8**: Good
- **AUC = 0.5**: Random guessing (useless)
- **AUC < 0.5**: Worse than random

**Intuitive meaning**: AUC = probability that model ranks a random positive example higher than a random negative example.

## How ROC is Generated

1. Model outputs probabilities for each sample
2. Generate thresholds (typically every unique probability + ∞ and 0)
3. **For each threshold**:
   - Classify: predict 1 if prob ≥ threshold, else 0
   - Calculate confusion matrix → get TP, FP, TN, FN (See [[Precision, Recall, and F1-Score]])
   - Calculate TPR = TP/(TP+FN) and FPR = FP/(FP+TN)
   - This gives ONE point (FPR, TPR) on the curve
4. Plot all points → connect them → ROC curve!

## Key Insights

- **Each threshold = one point on ROC curve**
- **Curve shows trade-offs**: Moving threshold changes both TPR and FPR
- **Top-left corner = ideal**: High TPR, low FPR
- **Diagonal line = random**: 50% AUC means no predictive power
- **Threshold-independent**: AUC evaluates ranking ability, not specific threshold

## When to Use ROC/AUC

**Good for**:
- Comparing different models
- Balanced classes or equal importance of both classes
- Threshold-independent evaluation
- Understanding trade-offs between TPR and FPR

**Caution**:
- Severe class imbalance (consider Precision-Recall curves instead)
- When false positives and false negatives have very different costs

Different use cases need different operating points on the ROC curve:

| Use Case | Priority | Threshold Choice |
|----------|----------|------------------|
| Medical screening | High TPR (catch all diseases) | Lower threshold (more aggressive) |
| Spam filter | Low FPR (few false alarms) | Higher threshold (more conservative) |
| Balanced | Equal TPR and FPR importance | Usually ~0.5 |

## Sample Example
![[Pasted image 20251114124012.png]]
## References
- [Fawcett (2006) - An Introduction to ROC Analysis](https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf)
- [Scikit-learn ROC Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Google ML Course - ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---
**Back to**: [[ML & AI Index]]
