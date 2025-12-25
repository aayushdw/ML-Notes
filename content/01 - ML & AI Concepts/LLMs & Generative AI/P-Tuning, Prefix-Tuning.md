# P-Tuning

## Overview
P-Tuning is an approach to get around the constraints / difficulty of "Prompt Tuning / Engineering".
Discrete prompting is an NP-hard search problem over a non-differentiable landscape. We are forcing a continuous model to be steered by discrete inputs.

P-Tuning shifts the approach from **Combinatorial Optimization** (Discrete) to **Continuous Optimization** (Differentiable).

## Key Mathematical Foundation

In standard interaction with an LLM, we treat the model as a function $M$ that takes a sequence of tokens from a fixed vocabulary $\mathcal{V}$.

Let an input sequence be $x$ and a target output be $y$.
We want to find a prompt $P$ (a sequence of tokens $[p_1, p_2, \dots, p_L]$) that maximizes the likelihood of $y$.
The Objective Function:
$$\hat{P} = \underset{P}{\text{argmax}} \sum_{(x,y) \in \mathcal{D}} \log P(y | x, P; \Theta)$$

The Constraint:
$$\forall p_i \in P, \quad p_i \in \mathcal{V}$$
#### **Why this is mathematically problematic:**
1. **Non-Differentiable:** The operation of selecting a token $p_i$ from vocabulary $\mathcal{V}$ is a discrete indexing operation. You cannot compute the gradient $\nabla_P \mathcal{L}$ because $P$ is not a continuous variable. The loss landscape is a series of step functions, not a smooth curve.
    
2. **Combinatorial Explosion:** If the vocabulary size is $|\mathcal{V}| \approx 50,000$ and you want a prompt of length $L=20$, the search space is $50,000^{20}$.
    
3. **Local Optima:** Discrete search methods (like genetic algorithms or reinforcement learning used in discrete prompt search) often get stuck in local optima because they cannot follow a gradient to the global minimum.

> **Note:** Approaches like **AutoPrompt**[AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts] attempted to solve this using gradient-guided search to find discrete tokens, but they are computationally expensive and often result in gibberish prompts (e.g., _"Horse staple battery correct"_ might trigger a specific behavior).


#### **The Visual Intuition**:
To understand the solution, look at first layer of the Transformer: the **Embedding Matrix** $E \in \mathbb{R}^{|\mathcal{V}| \times d}$, where $d$ is the hidden dimension.
Every discrete token $p_i$ corresponds strictly to a single row vector in $E$.
Imagine the continuous vector space $\mathbb{R}^d$. The "valid" English words occupy only tiny, specific points in this vast space.

- **Discrete Prompting:** You are forced to hop only between these specific points (words).
- **The Gap:** The optimal vector that triggers the model to solve your task might lie in the "empty space" _between_ the words "Translate" and "French".

Because natural language evolved for human communication, not for steering high-dimensional neural manifolds, **natural language is likely a suboptimal control language for LLMs.**

### **The Continuous Relaxation (P-Tuning Solution)**

P-Tuning applies a mathematical trick common in optimization called **Relaxation**. We relax the constraint that our prompt tokens must be integers mapping to rows in $E$.
Instead, we define the prompt $P$ as a sequence of **free** vectors:
$$P = [\theta_1, \theta_2, \dots, \theta_L]$$
Where each $\theta_i \in \mathbb{R}^d$.

The New Objective Function:
$$\hat{\theta} = \underset{\theta}{\text{argmax}} \sum_{(x,y) \in \mathcal{D}} \log P(y | x, \theta; \Theta_{fixed})$$
**Why this changes everything:**
1. Differentiability: Because $\theta$ operates in continuous space, the loss function is now fully differentiable with respect to $\theta$.
   We can use standard Backpropagation (SGD, AdamW) to optimize the prompt.
    
2. **Expressivity:** The continuous prompt space contains the discrete token space (since every word vector is a point in $\mathbb{R}^d$), but it also contains everything in between.
    - _Hypothesis:_ Continuous prompts can encode "instructions" that are semantically impossible to express in human language but perfectly interpretable by the model's attention mechanisms.


## Initialization Setback

Let's look at the input processing of the first Transformer block.
Standard (Discrete):
$$H_0 = [E(p_1), E(p_2), \dots, E(x_1), \dots]$$
The prompt vectors are frozen lookup values.

P-Tuning (Continuous):
$$H_0 = [\theta_1, \theta_2, \dots, E(x_1), \dots]$$
Here, $\theta_i$ are trainable parameters.

However, P-Tuning v1 recognized a specific optimization difficulty here.
If you initialize $\theta_i$ randomly (Gaussian noise), they are independent variables. But natural language is sequential; word $i$ depends on word $i-1$.

If we treat $\theta_1 \dots \theta_L$ as independent variables, the optimization landscape is too chaotic. The model struggles to find a "coherent" prompt because the parameters can move in opposing directions in the vector space.

**Potential Fix**: Very slow learning rates and extremely large models (10B+) to work. For normal scales, it is unstable.


## Reparameterization
P-Tuning introduces a function $f$ to **reparameterize** the prompt embeddings. Instead of training the embeddings directly, we train the parameters of a small neural network (the Encoder) that _generates_ the embeddings.



TODO (https://gemini.google.com/gem/efe5a0156c9d/a45ca66d5f6f5430)



---

**Progress**: 
- [ ] Read overview materials
- [ ] Understand key concepts
- [ ] Review mathematical foundations
- [ ] Study implementations
- [ ] Complete hands-on practice
- [ ] Can explain to others

**Status Options**: `not-started` | `in-progress` | `completed` | `review-needed`
**Difficulty Options**: `beginner` | `intermediate` | `advanced` | `expert`

---
**Back to**: [[ML & AI Index]]
