# A/B Testing (Online Experimentation)

## Overview
**A/B Testing** (Split Testing) is the gold standard for causal inference in product development. Unlike offline evaluation (Accuracy/F1), which proxies performance, A/B testing measures the actual impact on business metrics (Revenue, Retention, CTR) in a live environment.

## Key Ideas / Intuition
- **Randomization**: Randomly assigning users to groups (Control vs. Treatment) eliminates confounding variables (e.g., time of day, user demographics), ensuring that any difference in outcome is *caused* by the model.
- **Hypothesis Testing**: We are not just checking if $B > A$; we are checking if the difference is *statistically significant* or just noise.

## Core Concepts

### 1. Statistical Foundations
We test a **Null Hypothesis** ($H_0$) against an **Alternative Hypothesis** ($H_1$).

> [!TIP] **Intuition**
> * **$H_0$ (Null Hypothesis)**: The new model has *no effect*.
> * **Goal**: We assume $H_0$ is true until evidence (data) proves otherwise.
> * **P-Value**: The probability that the evidence appeared purely by coincidence.
>     * Low P-Value (< 0.05) → "It's extremely unlikely this happened by chance." → $H_1$ is true.
>     * High P-Value → "Evidence is weak." → Stick with $H_0$.

#### Errors
| Decision | Reality: $H_0$ is True (No Diff) | Reality: $H_0$ is False (Diff exists) |
| :--- | :--- | :--- |
| **Accept $H_0$** | Correct | **Type II Error ($\beta$)**<br>*(Letting a criminal go free)* |
| **Reject $H_0$** | **Type I Error ($\alpha$)**<br>*(Convicting an innocent person)* | Correct (Power) |

- **Significance Level ($\alpha$)**: Usually 0.05. We accept a 5% risk of a False Positive.
- **Power ($1 - \beta$)**: Usually 0.80. We want an 80% chance of catching a real improvement.

### 2. Metric Hierarchy
- **North Star Metric**: The ultimate long-term goal (e.g., Customer Lifetime Value). Hard to move in short tests.
- **Driver Metrics** (OEC - Overall Evaluation Criteria): Short-term proxies that correlate with North Star.
    - *Example*: CTR, Conversion Rate, Session Length.
- **Guardrail Metrics**: Constraints that must not be violated.
    - *Example*: Latency, Error Rate, Unsubscribe Rate.

### 3. Sample Size Calculation
The number of users $N$ required depends on the **Minimum Detectable Effect (MDE)** ($\delta$) and Variance ($\sigma^2$).
$$ N \approx 16 \frac{\sigma^2}{\delta^2} $$
- **Takeaway**: To detect a smaller improvement ($\delta \downarrow$), you need quadratically more users ($N \uparrow^2$).
- **Takeaway**: Reducing metric variance ($\sigma^2$) (e.g., using CUPED) allows for faster experiments.

---

## Common Pitfalls

### 1. Sample Ratio Mismatch (SRM)
If you target a 50/50 split but get 49/51, **STOP**.
- **Cause**: The treatment model might be crashing/slower for some users, causing them to drop out before the event is logged.
- **Result**: The groups are no longer comparable. Invalidates the test.

### 2. Peeking (P-hacking)
Checking the p-value every day and stopping as soon as $p < 0.05$.
- **Problem**: Increases False Positive rate to >30%.
- **Solution**: Fix sample size in advance or use **Sequential Testing** (SPRT).

### 3. Novelty Effect
Users click the new feature just because it looks different, not because it's better.
- **Solution**: Run the test longer (weeks) to let the novelty wear off.

### 4. Network Effects (Interference)
In two-sided marketplaces (Uber/Airbnb), treating one user affects others (e.g., Driver A gets a ride, Driver B doesn't).
- **Solution**: **Cluster Randomization** (split by city) or **Switchback Testing** (split by time windows).

---

## Comparison: A/B vs Bandits

| Feature | A/B Testing | Multi-Armed Bandits (MAB) |
| :--- | :--- | :--- |
| **Goal** | Statistical Significance (Knowledge) | Regret Minimization (Reward) |
| **Traffic Split** | Fixed (50/50) | Dynamic (shifts to winner) |
| **Best for** | Major UX/Model changes | Content optimization, Headlines, Ads |
| **Speed** | Slow (Safety first) | Fast (Optimization first) |

---

## Resources
- **Paper**: [Overlapping Experiment Infrastructure: More, Better, Faster Experimentation (Google)](https://research.google/pubs/pub36500/)
- **Book**: "Trustworthy Online Controlled Experiments" (Kohavi, Tang, Xu).


**Back to**: [[ML Deployment Patterns]]
