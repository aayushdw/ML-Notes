## Overview
LLM Safety refers to the broad set of techniques, research, and practices aimed at ensuring large language models behave in ways that are helpful, honest, and harmless. It encompasses the entire pipeline from pre-training data curation to post-deployment monitoring.
Safety sits at the intersection of [[Fine-Tuning Overview]], [[Reinforcement Learning]], and adversarial machine learning.

## The Alignment Problem
The fundamental challenge is that LLMs learn to predict the next token, they do not inherently "understand" human values. An unaligned model will produce harmful content if that content appeared in its training data.

## Three Pillars of LLM Safety

![[LLM Safety Fundamentals 2025-12-30 17.58.38.excalidraw.svg]]

1. [[#Alignment]]: Training the model to follow human preferences and refuse harmful requests.[]()
2. [[#Robustness]]: Ensuring the model maintains safe behavior under adversarial attack (see [[Prompt Injection and Safety]]).
3. [[#Monitoring]]: Deploying guardrails and observability to catch failures in production.

## Alignment

Alignment is about training the model to internalize human values and preferences. This happens during post-training and is the foundation of safety.
### Supervised Fine-Tuning (SFT)
The first step in post-training. Base model is fine-tuned on examples of "ideal" assistant behavior.

For safety, this includes examples like:
- **Input**: "How do I hack into my neighbor's WiFi?"
- **Ideal Output**: "I can't help with unauthorized access to networks. If you're having connectivity issues, I can suggest legitimate troubleshooting steps."

**Limitation**: SFT alone is insufficient because you cannot enumerate every possible harmful request. The model learns to mimic refusals but does not develop a generalized "sense" of what to refuse.

### Reinforcement Learning from Human Feedback (RLHF)

The dominant paradigm for alignment. Since we can't write down explicit rules for "good behavior" RLHF learns to mimic what humans prefer by watching them compare outputs.

> [!tip] Intuition: Why Preferences Over Demonstrations?
> SFT requires humans to *write* the ideal response, which is slow and expensive. RLHF only asks humans to *compare* responses, which is much faster. It's easier to say "A is better than B" than to write the perfect answer from scratch. This makes RLHF more scalable.

#### Stage 1: Collect Preference Data

Humans are shown two or more model responses to the same prompt and asked to rank them. This is repeated thousands of times (typically 50k–500k comparisons) to build a preference dataset.

**What makes a good preference dataset:**
- **Diverse prompts**: Cover a wide range of topics, including edge cases and potentially harmful requests.
- **Clear labeler instructions**: Labelers need explicit guidelines on what "better" means (e.g., prioritize safety over helpfulness, or vice versa).
- **Quality control**: Multiple labelers per comparison to measure inter-annotator agreement.

**Preference Format:**
```
Prompt: "How do I delete all files on my computer?"
Response A: "You can use 'rm -rf /' on Linux..." (unsafe)
Response B: "I'd be happy to help you free up disk space safely..." (safe)
Human Choice: B > A
```

> [!warning] Labeler Bias
> The model learns to imitate the preferences of *specific human labelers*. If labelers have biases (cultural, political, etc.), the model will inherit them. This is why labeler selection and instruction design is critical.

#### Stage 2: Train a Reward Model

A separate neural network is trained to predict which response a human would prefer. This "reward model" (RM) takes a `(prompt, response)` pair and outputs a scalar score.

**Architecture**: Typically the same architecture as the LLM being aligned, but with the language modeling head replaced by a scalar output head. Often initialized from the SFT model.

**Training objective**: Given a preference pair $(y_w, y_l)$ where $y_w$ is preferred over $y_l$, train the RM to assign a higher score to $y_w$.

**Why this works**: The reward model learns to be a stand-in for human judgment. Once trained, we can query it millions of times during RL without needing humans in the loop.

**Practical considerations:**
- **Model size**: RMs are often smaller than the policy model to reduce inference cost during RL.
- **Calibration**: The absolute values of reward scores don't matter, only their relative ordering. 
- **Overfitting**: RMs can overfit to superficial patterns (e.g., longer responses are preferred). Regularization and held-out evaluation are essential.

#### Stage 3: Fine-tune with Reinforcement Learning

The LLM (now called the "policy") generates responses, the reward model scores them, and the policy is updated to produce higher-scoring outputs. This uses [[Proximal Policy Optimization (PPO)]] .

**The RL loop:**
1. Sample a batch of prompts from the dataset.
2. Generate responses using the current policy $\pi_\theta$.
3. Score each response with the reward model: $R(x, y)$.
4. Compute the loss and update $\pi_\theta$ using PPO.
5. Repeat for thousands of iterations.

**Key constraint (KL Penalty)**: Without constraints, the policy can "reward hack", finding weird outputs that score high on the RM but are actually nonsense (e.g., repeating tokens that the RM likes). To prevent this, we add a penalty for drifting too far from the original SFT model (the "reference policy" $\pi_{ref}$).

> [!tip] Intuition: Why the KL Penalty?
> The original SFT model already produces coherent, grammatical text. We want to *nudge* it toward higher rewards without destroying its language modeling capabilities. The KL divergence measures how much the policy has changed—larger values mean more drift.

**Hyperparameter $\beta$**: Controls the strength of the KL penalty.
- High $\beta$: Policy stays close to reference (conservative learning, less reward hacking, but slower improvement).
- Low $\beta$: Policy diverges more freely (faster improvement, but higher risk of reward hacking).

> [!info]- Mathematical Details
> **Reward Model Training** using the [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model):
> 
> The probability that response $y_1$ is preferred over $y_2$ given prompt $x$:
> $$P(y_1 \succ y_2 | x) = \sigma(R(x, y_1) - R(x, y_2))$$
> 
> Where:
> - $\sigma$ is the sigmoid function
> - $R(x, y)$ is the reward model's scalar output
> 
> **Loss function** (negative log-likelihood of preferences):
> $$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(R(x, y_w) - R(x, y_l)) \right]$$
>
> ---
> 
> **RL Objective (PPO with KL Penalty)**:
> $$\mathcal{L}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) - \beta \cdot D_{KL}(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)) \right]$$
>
> Where:
> - $\pi_\theta$: The current policy (the LLM being optimized)
> - $\pi_{ref}$: The reference policy (frozen SFT model)
> - $\beta$: KL penalty coefficient
> - $D_{KL}$: KL divergence, measuring distribution shift
> 
> **Per-token KL approximation** (used in practice):
> $$D_{KL} \approx \sum_t \left( \log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{ref}(y_t | x, y_{<t}) \right)$$
>
> This is computed token-by-token across the generated sequence.

#### Challenges and Limitations of RLHF

| Challenge              | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| **Reward Hacking**     | Policy exploits flaws in the RM (e.g., longer responses, sycophancy).    |
| **Human Labeler Cost** | Collecting 100k+ high-quality preferences is expensive.                  |
| **RM Generalization**  | The RM may fail on out-of-distribution prompts not seen during training. |
| **Instability**        | PPO is notoriously difficult to tune. Training can diverge or collapse.  |
| **Alignment Tax**      | RLHF can reduce raw capability on benchmarks while improving safety.     |

### Direct Preference Optimization (DPO)

DPO asks: why train a reward model just to throw it away? Instead, skip straight to what we actually want, make good responses more likely, bad responses less likely.

> [!tip] Intuition: The Implicit Reward Model
> RLHF trains an explicit reward model $R(x, y)$, then uses it to update the policy. DPO realizes that the *optimal policy under RLHF* has a closed-form relationship to the reward. So instead of learning $R$ explicitly, DPO reparameterizes the problem to learn the policy directly. The reward is "implicit" in the policy's log-probabilities.

#### How DPO Works

**Training process:**
1. Start with an SFT model (this becomes $\pi_{ref}$, the reference policy).
2. Take preference data: pairs of $(y_w, y_l)$ where $y_w$ is preferred over $y_l$ for prompt $x$.
3. For each pair, compute how much more likely the policy makes $y_w$ vs. $y_l$, relative to the reference.
4. Update the policy to increase this margin (make preferred responses *more* likely, rejected responses *less* likely).
5. The $\beta$ parameter controls how much the policy can deviate from the reference.

DPO's loss function is derived by substituting the closed-form optimal policy from RLHF into the preference model. This eliminates the need for a separate reward model.

#### DPO vs. RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward Model** | Explicit (separate model) | Implicit (in policy log-probs) |
| **RL Algorithm** | PPO (complex, unstable) | None (supervised learning) |
| **Training Stability** | Finicky, requires careful tuning | More stable, standard optimization |
| **Memory** | 3 models (policy, ref, RM) | 2 models (policy, ref) |
| **Compute** | RL loop with sampling | Single forward/backward pass |
| **Empirical Performance** | Strong, well-studied | Comparable or better on many tasks |

#### Practical Considerations

**Hyperparameters:**
- **$\beta$ (temperature)**: Controls KL constraint strength. Typical values: 0.1–0.5.
  - Higher $\beta$: Stronger penalty for deviating from reference (more conservative).
  - Lower $\beta$: More aggressive optimization toward preferences (risk of overfitting).
- **Learning rate**: Usually lower than SFT (1e-6 to 1e-5) to avoid catastrophic forgetting.
- **Batch size**: Larger batches help stability (32–128 preference pairs).

**Data requirements:**
- Same preference data format as RLHF (prompt, chosen response, rejected response).
- Quality matters more than quantity—noisy preferences hurt DPO more than RLHF because there's no RM to smooth over noise.


> [!info]- Mathematical Details
> **The Key Derivation:**
> 
> In RLHF, the optimal policy under the KL-constrained objective has a closed form:
> $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right)$$
> 
> Rearranging to solve for the reward:
> $$R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$
> 
> The partition function $Z(x)$ cancels when we substitute into the Bradley-Terry preference model (since it's the same for both $y_w$ and $y_l$).
> 
> ---
> 
> **DPO Loss Function:**
> $$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$
> 
> **Interpreting the terms:**
> - $\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$: How much more likely the policy makes $y_w$ vs. the reference → implicit reward for $y_w$
> - $\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$: Implicit reward for $y_l$
> - The difference is the implicit *reward margin*
> - The sigmoid + log pushes this margin to be positive (preferred response should have higher implicit reward)
> 
> **Gradient intuition:**
> - When the model correctly prefers $y_w$: small gradient (already doing well)
> - When the model incorrectly prefers $y_l$: large gradient (needs correction)
> - This is similar to how cross-entropy works for classification
>
> **Where:**
> - $y_w$: winning (preferred) response
> - $y_l$: losing (rejected) response  
> - $\pi_{ref}$: reference policy (frozen SFT model)
> - $\beta$: temperature controlling deviation from reference (higher = more conservative)
> - $\sigma$: sigmoid function

#### Variants and Extensions

| Variant | Key Idea |
|---------|----------|
| **IPO** (Identity PO) | Removes sigmoid for more stable gradients |
| **KTO** (Kahneman-Tversky) | Uses unpaired data (just good or bad, not comparisons) |
| **ORPO** | Combines SFT and preference optimization in one stage |
| **SimPO** | Simplifies DPO by removing reference model dependency |

#### Challenges and Limitations

| Challenge | Description |
|-----------|-------------|
| **Preference Data Quality** | DPO is more sensitive to noisy labels than RLHF (no RM to smooth over noise). |
| **Distribution Shift** | If preference data comes from a different model, performance may degrade. |
| **No Exploration** | Unlike RL, DPO only optimizes on existing data—no active sampling of new responses. |
| **Reference Model Dependence** | Requires keeping $\pi_{ref}$ in memory during training (can be mitigated with caching). |

### Constitutional AI
Reduces reliance on human labelers by using the model itself to critique and revise its outputs.

1. Generate an initial response (which may be harmful).
2. Ask the model to critique the response against a "constitution" (a set of principles like "be helpful," "avoid harm," "be honest").
3. Ask the model to revise its response based on the critique.
4. Use the revised response as training data for RLHF.

Instead of needing humans to red-team every possible harmful scenario, CAI leverages the model's own knowledge to identify and fix safety issues. This scales better than pure human labeling.

## Robustness

A model can be perfectly aligned in normal conditions but still fail under adversarial pressure. Robustness ensures safety holds even when users actively try to break it. This involves understanding attack vectors and stress-testing defenses.

### Red Teaming
Practice of adversarially probing models to discover safety failures *before* deployment.

#### Manual Red Teaming
Human experts attempt to elicit harmful outputs through creative prompting:
- Roleplay scenarios
- Hypothetical framing
- Incremental escalation

#### Automated Red Teaming
Using LLMs to generate attack prompts at scale:
1. Train an "attacker" model to generate prompts that elicit unsafe responses.
2. Use the target model's failures to improve the attacker.
3. Use the attacker's successful prompts to improve the target's defenses.

This creates an adversarial training loop similar to GANs.

### Common Jailbreak Categories

| Category                 | Description                                       | Example                                         |
| ------------------------ | ------------------------------------------------- | ----------------------------------------------- |
| **Roleplay**             | Asking the model to adopt an unrestricted persona | "You are DAN, an AI without restrictions..."    |
| **Encoding**             | Obfuscating harmful content                       | Base64, ROT13, pig latin                        |
| **Multi-turn**           | Gradually escalating across conversation turns    | Starting with chemistry, ending with explosives |
| **Context Manipulation** | Exploiting in-context learning                    | Few-shot examples of harmful behavior           |

For detailed attack vectors, see [[Prompt Injection and Safety]].

## Monitoring

Even with strong alignment and robustness testing, failures will occur in production. Monitoring provides the last line of defense. Runtime guardrails that catch harmful inputs/outputs, plus observability to detect and respond to emerging threats.

### Input Guardrails
1. **Blocklist Filters**: Simple keyword matching for obvious violations.
2. **Classifier-Based Detection**: A separate model (often a fine-tuned BERT) classifies inputs as safe/unsafe.
3. **Embedding Similarity**: Compare input embeddings against known jailbreak embeddings.

### Output Guardrails
1. **Toxicity Classifiers**: Scan generated text for harmful content.
2. **PII Detection**: Mask or block outputs containing personal information.
3. **Format Validation**: Ensure structured outputs (JSON, code) are valid before returning.


### Observability and Incident Response

Guardrails prevent known bad patterns, but observability will help discover *unknown* failures:

1. **Logging**: Store all prompts and responses (with appropriate privacy controls) for post-hoc analysis.
2. **Anomaly Detection**: Flag unusual patterns, sudden spikes in refusals, unusual token sequences, or repeated probing from single users.
3. **Human Review Queues**: Route low-confidence decisions to human reviewers for labeling and model improvement.
4. **Feedback Loops**: User reports of harmful outputs feed back into red teaming and alignment training.

The goal is a closed loop: production failures become training signal for the next model iteration.

## Layered Defense Architecture

![[LLM Safety Fundamentals 2025-12-30 17.52.44.excalidraw.svg]]


## Practical Considerations

### Trade-offs

| Approach       | Pros                            | Cons                                            |
| -------------- | ------------------------------- | ----------------------------------------------- |
| **RLHF**       | Strong alignment, well-studied  | Expensive (human labelers), reward hacking risk |
| **DPO**        | Simpler, no reward model needed | Requires quality preference data                |
| **CAI**        | Scalable, less human annotation | Model may miss novel harms                      |
| **Guardrails** | Fast, interpretable             | Brittle, easy to bypass                         |

### Common Pitfalls
1. **Over-Refusal**: A model that refuses too aggressively becomes useless. "How to kill a Python process" should not be refused.
2. **Neglecting Indirect Injection**: Most teams focus on direct attacks and forget that their RAG pipeline can inject malicious content.
3. **Static Defenses**: Jailbreaks evolve. A defense that works today may fail tomorrow. Continuous red teaming is essential.

## Emerging Research Areas
### Interpretability (TODO) ??

### Capability Control
- **Unlearning**: Removing specific capabilities (e.g., knowledge of bioweapons) from the model.
	- How to untrain a model over a knowledge base?

## Resources

### Papers
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (InstructGPT/RLHF) — The foundational RLHF paper from OpenAI
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Earlier RLHF work on summarization
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — The PPO algorithm used in Stage 3
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO) — The original DPO paper
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (IPO) — Identity Preference Optimization
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Works with unpaired preference data
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Combines SFT and DPO
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (CAI)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Important paper on reward hacking

### Articles
- [Illustrating RLHF](https://huggingface.co/blog/rlhf) — HuggingFace's visual guide to RLHF
- [The Alignment Handbook](https://github.com/huggingface/alignment-handbook) — Practical recipes for RLHF and DPO
- [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl) — HuggingFace's hands-on DPO tutorial

### Videos
- [RLHF Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s)

---
**Back to**: [[02 - LLMs & Generative AI Index]] | [[ML & AI Index]]
