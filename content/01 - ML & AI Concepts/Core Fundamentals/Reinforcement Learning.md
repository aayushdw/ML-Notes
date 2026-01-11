## Overview

Reinforcement Learning (RL) is a learning paradigm where an **agent** learns to make decisions by interacting with an **environment**. Unlike [[Supervised Learning]] where we have labeled input-output pairs, or [[Unsupervised Learning]] where we discover patterns in unlabeled data, RL learns through **trial and error** by receiving **rewards** or **penalties** for actions taken.

RL optimizes for **long-term cumulative reward**, not immediate feedback. An agent might take a seemingly poor action now (like sacrificing a chess piece) if it leads to better outcomes later (winning the game).

## RL Mental Model

Think of training a dog: you don't show it examples of "correct" sits, you reward good behavior and ignore (or penalize) bad behavior. Over time, the dog learns which actions lead to treats.

![[Reinforcement Learning 2025-12-31 15.12.03.excalidraw.svg]]

### Exploration vs Exploitation
- **Exploitation**: Use current knowledge to maximize immediate reward
- **Exploration**: Try new actions to potentially discover better strategies

Too much exploitation → stuck in local optima, miss better solutions
Too much exploration → waste time on suboptimal actions

Common solution: **ε-greedy** strategy
- With probability $\epsilon$ : EXPLORE (random action)
- With probability $1-\epsilon$ : EXPLOIT (best known action)

## Mathematical Foundation

### Markov Decision Processes (MDPs)

RL problems are typically formalized as MDPs, defined by the tuple $(S, A, P, R, \gamma)$:

| Symbol | Name | Definition |
|--------|------|------------|
| $S$ | State space | Set of all possible states |
| $A$ | Action space | Set of all possible actions |
| $P(s' \mid s, a)$ | Transition function | Probability of transitioning to $s'$ given state $s$ and action $a$ |
| $R(s, a, s')$ | Reward function | Immediate reward for transition |
| $\gamma$ | Discount factor | $\gamma \in [0, 1]$, determines importance of future rewards |

**Markov Property**: Future states depend only on the current state, not on the history of how we got there:

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$$

### Return and Value Functions

**Return** $G_t$: The cumulative discounted reward from timestep $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Why discount? 
- Mathematically ensures the sum converges (for $\gamma < 1$)
- Reflects preference for immediate rewards over uncertain future ones
	- $\gamma = 0$: myopic (only care about immediate reward)
	- $\gamma \to 1$: far-sighted (care equally about all future rewards)

### State-Value vs. Action-Value Functions

These two functions answer different questions:

| Function | Question It Answers | Notation | Depends On |
|----------|--------------------| ---------|------------|
| **State-Value** $V^\pi(s)$ | "How good is it to **be in** state $s$?" | $V^\pi(s)$ | State only |
| **Action-Value** $Q^\pi(s, a)$ | "How good is it to **take action** $a$ in state $s$?" | $Q^\pi(s, a)$ | State AND action |

**State-Value Function** $V^\pi(s)$: Expected return starting from state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s\right]$$

**Action-Value Function** $Q^\pi(s, a)$: Expected return starting from state $s$, taking action $a$, then following $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

**Intuition**:
- $V(s)$ tells you: "If I land in this chess position and play optimally from here, what's my expected outcome?"
- $Q(s, a)$ tells you: "If I land in this chess position and make *this specific move*, then play optimally, what's my expected outcome?"

$Q$ is actionable because it helps you **compare** different actions directly. Given $Q^*(s, a)$ for all actions, you immediately know the optimal policy: just pick $\arg\max_a Q^*(s, a)$.

**When to use each**:
- **$V(s)$**: When you already have a policy and want to evaluate states (e.g., policy iteration, actor-critic critics)
- **$Q(s, a)$**: When you need to choose actions without an explicit policy (e.g., Q-Learning, DQN)

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \cdot Q^\pi(s, a)$$

i.e., The value of a state is the weighted average of action-values, weighted by the policy's action probabilities. If the policy says "I pick action $a_1$ 70% of the time and $a_2$ 30%," then $V(s) = 0.7 \cdot Q(s, a_1) + 0.3 \cdot Q(s, a_2)$.

### Bellman Equations

 Core idea: **the value of a state depends on the values of states you can reach from it**. This recursive structure is what makes RL tractable.

#### Bellman Expectation Equation

For a fixed policy $\pi$, the value of being in state $s$ can be broken down as:

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

1. $\pi(a|s)$: Probability of taking action $a$ in state $s$ (given by policy)
2. $P(s'|s,a)$: Probability of landing in state $s'$ after taking action $a$
3. $R(s,a,s')$: Immediate reward for that transition
4. $\gamma V^\pi(s')$: Discounted value of where we end up
5. Sum over all possible actions and outcomes → expected value

i.e., Value here = average of (immediate reward + discounted future value), considering all the actions I might take and all the places I might end up.

#### Bellman Optimality Equation

For the **optimal** policy $\pi^*$, we don't average over actions, we take the **best** one:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]$$


**Why this matters**: If we can solve for $Q^*$, the optimal policy falls out trivially:
$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

i.e., "Just pick the action with the highest Q-value."

#### How to Solve for $Q^*$

The Bellman equation tells us *what* $Q^*$ must satisfy, but how do we actually find it?

**1. Model-Based (Value Iteration)**: If you know $P(s'|s,a)$ and $R$, repeatedly apply the Bellman equation as an update:

$$Q_{k+1}(s,a) \leftarrow \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q_k(s', a') \right]$$

Keep iterating until $Q$ stops changing. This converges to $Q^*$.

**2. Model-Free (Q-Learning)**: If you don't know the dynamics, learn from experience:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

Each time you take action $a$, observe reward $r$ and next state $s'$, then nudge your estimate toward the observed target. See the Q-Learning section below for details.

### Policy Definitions

**Deterministic Policy**: $\pi: S \to A$ maps each state to exactly one action.

**Stochastic Policy**: $\pi(a|s) = P(A_t = a | S_t = s)$ gives probability of each action in each state.

The **optimal policy** $\pi^*$ satisfies:
$$V^{\pi^*}(s) \geq V^\pi(s) \quad \forall s \in S, \forall \pi$$

## Core Algorithms

![[Reinforcement Learning 2025-12-31 15.55.23.excalidraw.svg]]

### Value-Based Methods

#### Q-Learning (Off-Policy)

The most famous RL algorithm. Learns $Q^*$ directly **without** following the optimal policy during training.

**Update Rule**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \underbrace{\left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]}_{\text{TD error}}$$

**Breaking it down**:
- $\alpha$: Learning rate
- $r + \gamma \max_{a'} Q(s', a')$: **TD (Temporal Difference) target** (observed reward + best future value)
- $Q(s, a)$: Current estimate
- **TD error**: How wrong we were (target - estimate)

Each update nudges our estimate toward what we actually observed.

**Why "off-policy"?** We use $\max_{a'} Q(s', a')$ in the update (the *greedy* action), even if we didn't actually take the greedy action. Our exploration policy (e.g., ε-greedy) differs from what we're learning about (the optimal policy).

![[Reinforcement Learning 2025-12-31 16.18.32.excalidraw.svg]]


#### SARSA (On-Policy)

**S**tate-**A**ction-**R**eward-**S**tate-**A**ction: Uses the actual next action, not the best one.

**Update Rule**:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

**Key difference from Q-Learning**: Uses $Q(s_{t+1}, a_{t+1})$ where $a_{t+1}$ is the action actually selected (following current policy), not $\max_a Q$.

**On-policy**: Learning and behavior use the same policy. SARSA is more conservative in risky environments because it accounts for exploration in its updates.

![[Reinforcement Learning 2025-12-31 16.49.51.excalidraw.svg]]
See: https://www.youtube.com/watch?v=tbpBW5Yr44k&t=12s 
#### Deep Q-Network (DQN) (TODO)

Q-Learning fails for large state spaces (can't store table for $10^{170}$ chess positions). DQN approximates $Q(s,a)$ with a neural network $Q(s, a; \theta)$.

1. **Experience Replay**: Store transitions $(s, a, r, s')$ in a replay buffer. Sample random mini-batches to break correlation and improve data efficiency.

2. **Target Network**: Use a separate, slowly-updated network $Q(s, a; \theta^-)$ for the TD target:
$$L(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

This stabilizes training by preventing the target from changing too rapidly.

**DQN Training Loop**:
```
Initialize replay buffer D, Q-network θ, target network θ⁻ ← θ
For each episode:
    For each step:
        Select action a using ε-greedy from Q(s, ·; θ)
        Execute a, observe r, s'
        Store (s, a, r, s') in D
        Sample mini-batch from D
        Compute target: y = r + γ max_a' Q(s', a'; θ⁻)
        Gradient descent on (y - Q(s, a; θ))²
        Every C steps: θ⁻ ← θ
```

### Policy-Based Methods

Instead of learning values and deriving a policy, directly learn a parameterized policy $\pi_\theta(a|s)$.

**Why policy-based?**
- Can handle continuous action spaces naturally
- Can learn stochastic policies (useful when optimal behavior is stochastic)
- Often have better convergence properties

#### Policy Gradient Theorem

The objective is to maximize expected return:
$$J(\theta) = \mathbb{E}_{\pi_\theta}[G_0] = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right]$$

**Policy Gradient Theorem** (the fundamental result):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]$$

i.e. Increase probability of actions that lead to high returns. The gradient of $\log \pi$ points in the direction to increase that action's probability; scale by how good the action was.

#### REINFORCE Algorithm

A Monte Carlo policy gradient method: run a full episode, then update the policy based on actual returns observed.

**Update Rule**:
$$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$$

**Breaking it down**:
- $\alpha$: Learning rate
- $\gamma^t$: Discount factor for timestep $t$ (earlier actions weighted more)
- $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_{k+1}$: Actual return from timestep $t$ onwards
- $\nabla_\theta \log \pi_\theta(a_t | s_t)$: Direction to increase probability of action $a_t$

**How it works**:
1. Run a complete episode using current policy $\pi_\theta$
2. For each timestep $t$, compute the return $G_t$ (sum of discounted rewards from $t$ to end)
3. Update: if $G_t$ is high, push $\theta$ to make $a_t$ more likely; if $G_t$ is low, push $\theta$ to make $a_t$ less likely
4. Repeat for many episodes

**Why $\log \pi$ ?** The gradient $\nabla_\theta \log \pi_\theta(a|s)$ is the "score function." Multiplying by return gives us an unbiased estimate of the policy gradient. High return → increase action probability. Low return → decrease it.

---

**The Variance Problem**

REINFORCE has high variance because $G_t$ can vary wildly between episodes (different random trajectories lead to very different returns). This means:
- Noisy gradient estimates
- Slow, unstable learning
- Needs many samples to average out the noise

**Solution: Baseline Subtraction**

Subtract a baseline $b(s)$ from the return:
$$\theta \leftarrow \theta + \alpha \gamma^t (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t | s_t)$$

**Why this helps**:
- If $G_t > b(s_t)$: action was better than average → increase probability
- If $G_t < b(s_t)$: action was worse than average → decrease probability
- Centering around the baseline reduces the magnitude of updates, lowering variance

**Common baseline**: $b(s) = V(s)$, the state-value function. This is optimal in the sense that it minimizes variance without introducing bias.

**Key insight**: Subtracting a baseline that doesn't depend on the action preserves the expected gradient (unbiased) while reducing variance. This leads naturally to Actor-Critic methods, where we learn $V(s)$ alongside the policy.

### Actor-Critic Methods

Combine value-based and policy-based approaches:
- **Actor**: The policy $\pi_\theta(a|s)$ that selects actions
- **Critic**: A value function $V_\phi(s)$ or $Q_\phi(s,a)$ that evaluates how good actions are

The critic reduces variance by replacing high-variance Monte Carlo returns with lower-variance value estimates.

**Advantage Function**:
$$A(s, a) = Q(s, a) - V(s)$$

Measures how much better action $a$ is compared to the average action from state $s$.

**Actor Update** (using advantage):
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)$$

**Critic Update** (minimize TD error):
$$\phi \leftarrow \phi - \beta \nabla_\phi (r + \gamma V_\phi(s') - V_\phi(s))^2$$

#### [[Proximal Policy Optimization (PPO)]]
Refer to linked note.

---

## Types of RL Problems

### Episodic vs. Continuing Tasks

| Aspect        | Episodic                                  | Continuing                                     |
| ------------- | ----------------------------------------- | ---------------------------------------------- |
| **Structure** | Clear start and end                       | Runs forever                                   |
| **Examples**  | Games, robot navigation                   | Stock trading, HVAC control                    |
| **Return**    | $G_t = \sum_{k=0}^{T} \gamma^k R_{t+k+1}$ | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ |
| **Gamma**     | Can use $\gamma = 1$                      | Need $\gamma < 1$ for convergence              |

### Model-Based vs. Model-Free

**Model-Free**: Learn policy or value function directly from experience. No explicit model of environment dynamics.
- Pro: Works when dynamics are unknown or complex
- Con: Sample inefficient (need lots of experience)
- Examples: Q-Learning, SARSA, Policy Gradient, PPO

**Model-Based**: Learn a model of the environment (transition and reward functions), then use it for planning.
- Pro: Sample efficient (can simulate experience)
- Con: Model errors can compound; complex dynamics are hard to model
- Example: Dyna-Q (combines learning with planning)

### On-Policy vs. Off-Policy

**On-Policy**: Learn about the policy currently being used for decisions.
- Must use fresh experience from current policy
- Examples: SARSA, REINFORCE, A2C/A3C, PPO

**Off-Policy**: Learn about a different policy than the one generating experience.
- Can reuse old experience (replay buffers)
- Examples: Q-Learning, DQN, DDPG, SAC

---

## Practical Applications

### When to Use Reinforcement Learning

**Good fit**:
- Sequential decision-making problems
- Clear reward signal (even if sparse)
- Ability to simulate or interact repeatedly with environment
- When optimal strategy is unknown or too complex to hand-code

**Examples**:
- Game playing (Go, Atari, Dota 2, StarCraft)
- Robotics and control (manipulation, locomotion)
- Recommendation systems (personalized content)
- Resource management (data centers, traffic control)
- LLM alignment ([[LLM Safety Fundamentals|RLHF for LLMs]])


### Pitfalls

1. **Reward Hacking**: Agent finds unintended ways to maximize reward
- Delete tests to ensure all tests pass? Sure.

2. **Sparse Rewards**: Agent receives feedback too rarely to learn
- Reward only for winning a chess game

3. **Sample Inefficiency**: Deep RL often needs millions of environment steps
   - Solutions: Model-based methods, better exploration, transfer learning

4. **Hyperparameter Sensitivity**: RL algorithms are notoriously finicky
   - Learning rate, discount factor, exploration schedule all matter greatly
   - Solutions: Maybe use PPO?

5. **Non-Stationarity**: In multi-agent settings, other agents are also learning
   - The "environment" keeps changing
   - Solutions: Self-play, population-based training

## More Advanced Topics (TODO sometime later??)

### Multi-Agent RL
Multiple agents learning simultaneously in shared environment. Challenges: non-stationarity, credit assignment among agents, emergent behavior.

### Hierarchical RL
Learn policies at multiple levels of abstraction. High-level policy selects goals, low-level policies achieve them. Helps with long-horizon problems.

### Inverse RL
Given demonstrations from an expert, infer the reward function they were optimizing. Useful when rewards are hard to specify but easy to demonstrate.

### Offline RL (Batch RL)
Learn from a fixed dataset of experience without further environment interaction. Important for real-world applications where online learning is expensive or dangerous.

### Safe RL
Learning while satisfying safety constraints. Critical for robotics and autonomous vehicles where exploration cannot be unconstrained.

---

## Resources

### Papers
- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Human-level control through deep RL (Nature DQN)](https://www.nature.com/articles/nature14236)

### Articles & Tutorials
- [OpenAI Spinning Up](https://spinningup.openai.com/) - Excellent introduction to deep RL
- [Lilian Weng's RL Blog Posts](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/)
- [David Silver's RL Course (DeepMind)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- [Pieter Abbeel's Deep RL Course (Berkeley)](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc)
- https://www.youtube.com/watch?v=tbpBW5Yr44k&t=12s 

### Code & Libraries
- [OpenAI Gym / Gymnasium](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) 
- [RLlib (Ray)](https://docs.ray.io/en/latest/rllib/index.html)

---
**Back to**: [[01 - Core Fundamentals Index]]
