## Overview
P-Tuning is an approach to get around the constraints / difficulty of Prompt Tuning.
Discrete prompting is an NP-hard search problem over a non-differentiable landscape. We are forcing a continuous model to be steered by discrete inputs.

P-Tuning shifts the approach from **Combinatorial Optimization** (Discrete) to **Continuous Optimization** (Differentiable).

## Key Mathematical Foundation

In standard interaction with an LLM, we treat the model as a function $M$ that takes a sequence of tokens from a fixed vocabulary $\mathcal{V}$.

Let an input sequence be $x$ and a target output be $y$.
We want to find a prompt $P$ (say a sequence of tokens $[p_1, p_2, \dots, p_L]$) that maximizes the likelihood of $y$.
The Objective Function:

$$\hat{P} = \underset{P}{\text{argmax}} \sum_{(x,y) \in \mathcal{D}} \log P(y | x, P; \Theta)$$

$$\forall p_i \in P, \quad p_i \in \mathcal{V}$$
#### **Why this is problematic:**
1. **Non-Differentiable:** The operation of selecting a token $p_i$ from vocabulary $\mathcal{V}$ is a discrete indexing operation. You cannot compute the gradient $\nabla_P \mathcal{L}$ because $P$ is not a continuous variable. The loss landscape is a series of step functions, not a smooth curve.
    
2. **Combinatorial Explosion:** If the vocabulary size is $|\mathcal{V}| \approx 50,000$ and you want a prompt of length $L=20$, the search space is $50,000^{20}$.
    
3. **Local Optima:** Discrete search methods (like genetic algorithms or reinforcement learning used in discrete prompt search) often get stuck in local optima because they cannot follow a gradient to the global minimum.

> **Note:** Approaches like **AutoPrompt**[AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts] attempted to solve this using gradient-guided search to find discrete tokens, but they are computationally expensive and can often result in gibberish prompts (e.g., _"Horse staple battery correct"_ might trigger a specific behavior).


#### **The Visual Intuition**:
To understand the solution, look at first layer of the Transformer: the **Embedding Matrix** $E \in \mathbb{R}^{|\mathcal{V}| \times d}$, where $d$ is the hidden dimension.
Every discrete token $p_i$ corresponds strictly to a single row vector in $E$.
Imagine the continuous vector space $\mathbb{R}^d$. The "valid" English words occupy only tiny, specific points in this vast space.

- **Discrete Prompting:** You are forced to hop only between these specific points (words).
- **The Gap:** The optimal vector that triggers the model to solve your task might lie in the "empty space" _between_ the words "Translate" and "French".

Because natural language evolved for human communication, not for steering high-dimensional neural manifolds, **natural language is likely a suboptimal control language for LLMs.**

### Continuous Relaxation (P-Tuning Solution)

P-Tuning applies a mathematical trick common in optimization called [Relaxation](https://en.wikipedia.org/wiki/Relaxation_(approximation)). We relax the constraint that our prompt tokens must be integers mapping to rows in $E$.
Instead, we define the prompt $P$ as a sequence of **free** vectors:

$$P = [\theta_1, \theta_2, \dots, \theta_L]$$
Where each $\theta_i \in \mathbb{R}^d$.

New Objective Function:

$$\hat{\theta} = \underset{\theta}{\text{argmax}} \sum_{(x,y) \in \mathcal{D}} \log P(y | x, \theta; \Theta_{fixed})$$

This results in:
1. Differentiability: Because $\theta$ operates in continuous space, the loss function is now fully differentiable with respect to $\theta$.
   We can use standard Backpropagation (SGD, AdamW) to optimize the prompt.
    
2. Expressivity: The continuous prompt space contains the discrete token space (since every word vector is a point in $\mathbb{R}^d$), but it also contains everything in between.
    - _Hypothesis:_ Continuous prompts can encode "instructions" that are semantically impossible to express in human language but perfectly interpretable by the model's attention mechanisms.


## Initialization Setback

Let's look at the input processing of the first Transformer block.
Standard (Discrete):

$$H_0 = [E(p_1), E(p_2), \dots, E(x_1), \dots]$$
The prompt vectors are frozen lookup values.

P-Tuning (Continuous):

$$H_0 = [\theta_1, \theta_2, \dots, E(x_1), \dots]$$
Here, $\theta_i$ are trainable parameters.

However, P-Tuning _v1_ recognized a specific optimization difficulty here.
If you initialize $\theta_i$ randomly (Gaussian noise), they are independent variables. But natural language is sequential; word $i$ depends on word $i-1$.

If we treat $\theta_1 \dots \theta_L$ as independent variables, the optimization landscape is too chaotic. The model struggles to find a "coherent" prompt because the parameters can move in opposing directions in the vector space.

**Potential Fix**: Very slow learning rates and extremely large models (10B+) to work. For normal scales, it is unstable.


## Reparameterization
P-Tuning introduces a function $f$ to **reparameterize** the prompt embeddings. Instead of training the embeddings directly, we train the parameters of a small neural network (the Encoder) that _generates_ the embeddings.

$$\theta_i = f(i; \phi)$$

Where $f$ is typically a small LSTM or MLP, and $\phi$ are its parameters.

This injects an **inductive bias of sequential coherence**. Because $f$ processes the position index $i$ through recurrent or dense layers, the output embeddings $\theta_1, \theta_2, \dots$ are no longer independent. They share structure through the shared parameters $\phi$.

The training now optimizes $\phi$, not $\theta$ directly:

$$\hat{\phi} = \underset{\phi}{\text{argmax}} \sum_{(x,y) \in \mathcal{D}} \log P(y | x, P_\phi; \Theta_{fixed})$$

Where $P_\phi = [f(1; \phi), f(2; \phi), \dots, f(L; \phi)]$ is the soft prompt generated by the encoder.

At inference time, we can discard the encoder $f$ and just use the generated embeddings $\theta_i = f(i; \phi^*)$ directly.


## Prefix-Tuning

Prefix-Tuning was developed independently around the same time as P-Tuning, with a different approach to the same problem. While P-Tuning focuses on NLU tasks and applies soft prompts only at the input layer, Prefix-Tuning targets **generation tasks** and applies prompts more aggressively throughout the model.

Prefix-Tuning prepends trainable "prefix" vectors to the **Key and Value matrices at every Transformer layer**.

For each layer $l$, we define prefix matrices:

$$P_K^{(l)}, P_V^{(l)} \in \mathbb{R}^{L_{prefix} \times d}$$

The attention computation becomes:

$$\text{Attention}(Q, [P_K; K], [P_V; V])$$

$[\cdot ; \cdot]$ denotes concatenation along the sequence dimension.

This results in:
- **Deeper Steering**: P-Tuning only influences the model at the first layer. Prefix-Tuning provides "control knobs" at every layer, allowing finer-grained steering of internal representations.
- **Same Reparameterization Trick**: Prefix-Tuning also uses an MLP to generate the prefix vectors during training, then discards it at inference.

## Comparison

| Aspect                    | Prompt Tuning            | P-Tuning               | Prefix-Tuning                                    |
| ------------------------- | ------------------------ | ---------------------- | ------------------------------------------------ |
| Where applied             | Input embeddings only    | Input embeddings only  | Every Transformer layer (K, V)                   |
| Reparameterization        | No (direct optimization) | Yes (LSTM/MLP encoder) | Yes (MLP encoder)                                |
| Parameter count           | $L \times d$             | Encoder params         | $2 \times L_{layers} \times L_{prefix} \times d$ |
| Stability on small models | Poor                     | Better                 | Best                                             |

## P-Tuning v2: Fixing the Model Size Limitation

P-Tuning (v1) has a critical limitation: **performance degrades significantly on smaller models** (under 10B parameters).

### Why Small Models Struggle

When soft prompts are applied only at the input embedding layer, their influence must propagate through all subsequent layers to affect the output. In smaller models:

1. **Limited Capacity**: Fewer parameters means the model has less capacity to "interpret" and propagate the soft prompt signal through its layers
2. **Signal Dilution**: The prompt's steering effect gets diluted as it passes through each layer, and smaller models have less redundancy to preserve this signal
3. **Optimization Difficulty**: The gradient path from loss back to the input-level soft prompts is long, making optimization harder

Empirically, P-Tuning v1 matches fine-tuning on 10B+ models but falls behind on models in the 300M-2B range.

### P-Tuning v2 Solution: Deep Prompt Tuning

P-Tuning v2 applies trainable prefixes to **every layer**, not just the input:

$$H^{(l)} = \text{Attention}(Q^{(l)}, [P_K^{(l)}; K^{(l)}], [P_V^{(l)}; V^{(l)}])$$

This provides:
- **Direct influence at each layer**: No need for the prompt signal to propagate through the entire network
- **Shorter gradient paths**: Each layer's prefix receives gradients directly from nearby computations
- **More parameters where they matter**: Instead of concentrating all trainable parameters at the input, they are distributed throughout the model

With this change, P-Tuning v2 matches fine-tuning performance even on 330M parameter models across NLU benchmarks (SuperGLUE, NER, QA).


## Practical Considerations

**When to Use:**
- To adapt a frozen LLM to a specific task without full fine-tuning
- Compute/memory budget is limited (only soft prompt parameters are updated)
- To maintain a single base model with multiple task-specific "heads" (just swap prompts)

**When NOT to Use:**
- The task requires significant deviation from pre-trained knowledge
- You have sufficient resources for [[LoRA (Low-Rank Adaptation)]] or full fine-tuning (typically more performant)


## Resources
- [GPT Understands, Too (P-Tuning)](https://arxiv.org/abs/2103.10385)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
	![[Screenshot 2026-01-08 at 1.46.05 AM.png]]
	P-Tuning achieved accuracy comparable to Adapter methods.
	
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning](https://arxiv.org/abs/2110.07602)



---
**Back to**: [[ML & AI Index]]
