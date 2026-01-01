## Overview

Proximal Policy Optimization (PPO) is a policy gradient algorithm that has become one of the most popular and practical deep [[Reinforcement Learning|RL]] algorithms. Developed by OpenAI in 2017, PPO addresses a fundamental challenge: policy gradient methods can be unstable when updates are too large, but being overly conservative wastes samples.

PPO strikes a balance by using a "clipped" objective function that prevents the policy from changing too drastically in any single update. It achieves competitive performance with algorithms like TRPO (Trust Region Policy Optimization) while being significantly simpler to implement and tune.

PPO is the algorithm behind the core RL algorithm used in [[LLM Safety Fundamentals#Reinforcement Learning from Human Feedback (RLHF)|RLHF for LLMs]].

## The Problem PPO Solves

Standard policy gradient methods (like [[Reinforcement Learning#REINFORCE Algorithm|REINFORCE]]) update the policy by taking gradient steps proportional to the return:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}$$

The problem: **how big should the step be?**

- **Too small steps** → Slow learning, wasted samples
- **Too large steps** → Policy collapses (performance tanks and may never recover)

Policy performance is extremely sensitive to parameter changes. A seemingly small $\theta$ update can dramatically shift action probabilities, leading to:
1. Collecting bad data with the broken policy
2. Using that bad data to make further bad updates
3. Catastrophic performance spiral

## Core Intuition: "Don't Change Too Much"

PPO's directly limits how much the *policy behavior* can change, not just the parameter change. It's done by:

1. **Tracking the probability ratio**: How much more/less likely is this action under the new policy vs the old?
2. **Clipping**: If the ratio gets too far from 1.0, stop the gradient signal

When advantage $A > 0$ (good action): Allow increases up to $1+\epsilon$, then clip.
When advantage $A < 0$ (bad action): Allow decreases down to $1-\epsilon$, then clip.

## Mathematical Foundation

### Policy Gradient Recap

The standard policy gradient objective is:

$$L^{PG}(\theta) = \mathbb{E}_t \left[ \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]$$

This has high variance and no mechanism to prevent large policy changes.

### The Probability Ratio

PPO introduces the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

This measures how much the new policy differs from the old one for the specific action taken:

| $r_t(\theta)$ | Meaning                                        |
| ------------- | ---------------------------------------------- |
| $r_t = 1$     | Action equally likely under old and new policy |
| $r_t > 1$     | New policy makes this action *more* likely     |
| $r_t < 1$     | New policy makes this action *less* likely     |


### PPO-Clip Objective

The core PPO objective (clipped version):

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

1. $r_t(\theta) \hat{A}_t$: The unclipped objective (like standard policy gradient with importance sampling)
2. $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$: Constrains the ratio to $[1-\epsilon, 1+\epsilon]$
3. $\min(\cdot, \cdot)$: Takes the more pessimistic (lower) estimate

**Why the minimum?**

The clipping is asymmetric depending on whether the advantage is positive or negative:

| Scenario | Advantage | Effect |
|----------|-----------|--------|
| Good action, ratio increases | $\hat{A}_t > 0$ | Cap benefit at $1+\epsilon$ |
| Good action, ratio decreases | $\hat{A}_t > 0$ | No clipping (allow decrease) |
| Bad action, ratio decreases | $\hat{A}_t < 0$ | Cap penalty at $1-\epsilon$ |
| Bad action, ratio increases | $\hat{A}_t < 0$ | No clipping (allow increase, reducing bad action) |

**Intuition**: We want to be conservative about changes that could hurt us, but permissive about changes that undo previous mistakes.

### The Full PPO Objective

In practice, PPO combines multiple objectives:

$$L^{PPO}(\theta) = \mathbb{E}_t \left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

Where:
- $L^{CLIP}$: The clipped surrogate objective (policy improvement)
- $L^{VF} = (V_\theta(s_t) - V_t^{target})^2$: Value function loss (critic training)
- $S[\pi_\theta]$: Entropy bonus (encourages exploration)
- $c_1, c_2$: Coefficients (typically $c_1 = 0.5$, $c_2 = 0.01$)

The entropy bonus $S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$ prevents the policy from becoming too deterministic too quickly.

### Generalized Advantage Estimation (GAE)

PPO typically uses GAE to compute advantages, which interpolates between high-bias/low-variance (TD) and low-bias/high-variance (Monte Carlo) estimates:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

The $\lambda$ parameter controls the trade-off:
- $\lambda = 0$: Pure TD (one-step), high bias, low variance
- $\lambda = 1$: Pure Monte Carlo, low bias, high variance
- $\lambda \approx 0.95$: Common default, good balance

## PPO Algorithm

```
Initialize policy network πθ, value network Vφ
for iteration = 1, 2, ... do
    // COLLECT DATA
    for actor = 1, ..., N do
        Run policy πθ_old in environment for T timesteps
        Collect trajectories {s, a, r, s'}
    end for
    
    // COMPUTE ADVANTAGES
    Compute rewards-to-go R̂t
    Compute advantage estimates Ât using GAE
    
    // OPTIMIZE (multiple epochs on same data!)
    for epoch = 1, ..., K do
        for minibatch in shuffle(collected_data) do
            // Policy update
            Compute r(θ) = πθ(a|s) / πθ_old(a|s)
            L_CLIP = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]
            
            // Value update
            L_VF = E[(Vφ(s) - R̂)²]
            
            // Entropy bonus
            S = E[entropy(πθ)]
            
            // Combined update
            Maximize L_CLIP - c1*L_VF + c2*S
        end for
    end for
    
    θ_old ← θ
end for
```

## Practical Application

### When to Use PPO

**Good fit**:
- Continuous or discrete action spaces
- When sample efficiency isn't critical but stability is
- When you want a reliable, well-understood baseline
- Multi-task and transfer learning scenarios
- RLHF for language models

**Not ideal for**:
- Extremely sample-constrained settings
- Very simple problems where vanilla policy gradient suffices
- When you need guaranteed convergence properties (PPO is empirically stable but has fewer theoretical guarantees than TRPO)

### Common Pitfalls

1. **Learning rate too high**: Policy can still destabilize despite clipping. Start conservative.

2. **Not normalizing advantages**: Advantages should be standardized (zero mean, unit variance) per batch for stable training.

3. **Value function divergence**: If the critic is poorly trained, advantage estimates become unreliable. Ensure adequate value function training.

4. **Too few epochs**: PPO's power comes from reusing data. Too few epochs waste samples.

5. **Too many epochs**: Overfitting to old data. The clipping helps, but there's still a limit. 10 epochs is usually the upper bound.

6. **Reward scaling issues**: Very large or very small rewards can cause numerical problems. Normalize if needed.

## Resources

### Papers
- [Proximal Policy Optimization Algorithms (Original Paper)](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438)
- [Emergent Complexity from Multi-Agent Competition (OpenAI)](https://arxiv.org/abs/1710.03748)

### Articles & Tutorials
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Lilian Weng - Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [HuggingFace Deep RL Course - PPO](https://huggingface.co/learn/deep-rl-course/unit8/introduction)
- [Arxiv Insights - PPO Explained](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [Pieter Abbeel - Policy Gradients and PPO](https://www.youtube.com/watch?v=y3oqOjHilio)
- [Stable Baselines3 - PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [CleanRL - PPO Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
- [SpinningUp - PPO Implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo)

---
**Back to**: [[Reinforcement Learning]] | [[01 - Core Fundamentals Index]]
