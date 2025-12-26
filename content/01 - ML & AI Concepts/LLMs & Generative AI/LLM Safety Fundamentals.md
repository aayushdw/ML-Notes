## Overview
LLM Safety refers to the broad set of techniques, research, and practices aimed at ensuring large language models behave in ways that are helpful, honest, and harmless. This is not just about filtering bad words—it encompasses the entire pipeline from pre-training data curation to post-deployment monitoring. Safety sits at the intersection of [[Fine-Tuning Overview]], [[Reinforcement Learning]], and adversarial machine learning.

### The Alignment Problem
The fundamental challenge is that LLMs learn to predict the next token—they do not inherently "understand" human values. An unaligned model will produce harmful content if that content appeared in its training data.

### Three Pillars of LLM Safety

```
                    LLM Safety
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   Alignment      Robustness       Monitoring
   (Training)     (Attacks)       (Deployment)
        │               │               │
     RLHF, CAI      Red Teaming     Guardrails
     DPO, SFT      Jailbreaks       Logging
```

1. Alignment: Training the model to follow human preferences and refuse harmful requests.
2. Robustness: Ensuring the model maintains safe behavior under adversarial attack (see [[Prompt Injection and Safety]]).
3. Monitoring: Deploying guardrails and observability to catch failures in production.

## Alignment Training Methods

### Supervised Fine-Tuning (SFT)
The first step in post-training. Base model is fine-tuned on examples of "ideal" assistant behavior.

For safety, this includes examples like:
- **Input**: "How do I hack into my neighbor's WiFi?"
- **Ideal Output**: "I can't help with unauthorized access to networks. If you're having connectivity issues, I can suggest legitimate troubleshooting steps."

**Limitation**: SFT alone is insufficient because you cannot enumerate every possible harmful request. The model learns to mimic refusals but does not develop a generalized "sense" of what to refuse.

### Reinforcement Learning from Human Feedback (RLHF)

Dominant paradigm for alignment.
#### Stage 1: Collect Comparison Data
Human labelers are shown multiple model outputs for the same prompt and rank them from best to worst.

#### Stage 2: Train a Reward Model
A separate neural network learns to predict the human ranking. Given a prompt $x$ and response $y$, the reward model outputs a scalar score $R(x, y)$.

The reward model is trained using the Bradley-Terry preference model:

$$P(y_1 \succ y_2 | x) = \sigma(R(x, y_1) - R(x, y_2))$$

Where:
- $y_1 \succ y_2$ means "response 1 is preferred over response 2"
- $\sigma$ is the sigmoid function
- The loss is binary cross-entropy over the preference pairs

#### Stage 3: Optimize the Policy with RL
The LLM (the "policy" $\pi_\theta$) is fine-tuned to maximize the reward model's score using Proximal Policy Optimization (PPO).

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ R(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

Where:
- $R(x, y)$ is the reward for generating response $y$ to prompt $x$
- $D_{KL}(\pi_\theta \| \pi_{ref})$ is the KL divergence from the reference (original SFT) policy
- $\beta$ controls the strength of the KL penalty

**Why the KL penalty?** Without it, the model would "reward hack"—finding degenerate outputs that score high on the reward model but are actually nonsense. The KL term keeps the model close to its original behavior.

### Direct Preference Optimization (DPO)

DPO is a simpler alternative to RLHF that skips the reward model entirely. Instead of training a reward model and then doing RL, DPO directly optimizes the policy on preference pairs.

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Where:
- $y_w$ is the "winning" (preferred) response
- $y_l$ is the "losing" (rejected) response
- $\pi_{ref}$ is the reference policy (original SFT model)
- $\beta$ controls how much the model can deviate from the reference

DPO increases the probability of preferred responses while decreasing the probability of rejected responses, all while staying close to the reference model. It is mathematically equivalent to RLHF under certain assumptions but is much easier to implement (no PPO, no reward model).

### Constitutional AI
Reduces reliance on human labelers by using the model itself to critique and revise its outputs.

1. Generate an initial response (which may be harmful).
2. Ask the model to critique the response against a "constitution" (a set of principles like "be helpful," "avoid harm," "be honest").
3. Ask the model to revise its response based on the critique.
4. Use the revised response as training data for RLHF.

Instead of needing humans to red-team every possible harmful scenario, CAI leverages the model's own knowledge to identify and fix safety issues. This scales better than pure human labeling.

## Red Teaming
Practice of adversarially probing models to discover safety failures *before* deployment.

### Manual Red Teaming
Human experts attempt to elicit harmful outputs through creative prompting:
- Roleplay scenarios
- Hypothetical framing
- Incremental escalation 
### Automated Red Teaming
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

## Guardrails and Production Safety

Alignment training is necessary but not sufficient. Production systems need runtime defenses.

### Input Guardrails
1. **Blocklist Filters**: Simple keyword matching for obvious violations.
2. **Classifier-Based Detection**: A separate model (often a fine-tuned BERT) classifies inputs as safe/unsafe.
3. **Embedding Similarity**: Compare input embeddings against known jailbreak embeddings.

### Output Guardrails
1. **Toxicity Classifiers**: Scan generated text for harmful content.
2. **PII Detection**: Mask or block outputs containing personal information.
3. **Format Validation**: Ensure structured outputs (JSON, code) are valid before returning.

### Layered Defense Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│  Input Filter   │ ← Blocklist, Classifier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Main LLM      │ ← Aligned via RLHF/DPO
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Filter  │ ← Toxicity, PII Scan
└────────┬────────┘
         │
         ▼
    User Output
```

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
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (InstructGPT/RLHF)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (CAI)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
### Videos
- [RLHF Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s)

---
**Back to**: [[02 - LLMs & Generative AI Index]] | [[ML & AI Index]]
