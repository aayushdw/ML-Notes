## Overview
Most popular PEFT technique. (See [[Fine-Tuning Overview]])
- **Concept:** Instead of updating a massive weight matrix $W$, LoRA injects two small matrices, $A$ and $B$, next to it.
	- The new output is $h = W_0x + BAx$.
	    - $W_0$ is frozen (the original model).
	    - $A$ and $B$ are trainable "adapter" matrices.

- **Result:** You can fine-tune a 70B model on a single consumer GPU because you are only training <1% of the parameters.

![[LoRA (Low-Rank Adaptation) 2025-12-30 22.06.21.excalidraw.svg]]

## Intuition

### Low Intrinsic Rank Hypothesis
When we fine-tune a massive model (e.g., 70 Billion parameters), we are updating the weights to adapt to a new task. The authors of the LoRA paper hypothesized that **weight updates are not random; they are highly correlated.**
If a model needs to learn to "speak like a pirate," it doesn't need to change every single neuron independently. The changes required to the weight matrices actually reside in a lower-dimensional space. This is called the **low intrinsic rank** hypothesis.

## Mathematical Foundation
In a standard dense layer of a neural network, you have a pre-trained weight matrix $W_0$.

In Full Fine-Tuning, you calculate a weight update $\Delta W$ that is the same size as $W_0$.

$$W_{new} = W_0 + \Delta W$$
In LoRA, we freeze $W_0$. We do not calculate $\Delta W$ directly. Instead, we decompose $\Delta W$ into two significantly smaller matrices, $A$ and $B$.

- **$W_0$ (Frozen):** The original model weights
- **$A$ (Trainable):** The "down-projection" matrix
- **$B$ (Trainable):** The "up-projection" matrix

The input $x$ passes through both paths effectively:

$$h = W_0x + BAx$$
### Compression
Let's assume our base layer $W_0$ has dimensions $d \times k$ (where $d$ is the output dimension and $k$ is the input dimension).
In standard fine-tuning, the update matrix $\Delta W$ must also be $d \times k$.

In LoRA, we choose a **Rank ($r$)**. This is a hyperparameter you control (typically very small, e.g., 8, 16, or 64).

The adapter matrices are:
1. **Matrix $B$:** Dimensions $d \times r$
2. **Matrix $A$:** Dimensions $r \times k$

When we multiply $B \times A$, the resulting dimension is $(d \times r) \times (r \times k) = d \times k$.

Therefore, the product $BA$ has the exact same shape as the original weights $W_0$, allowing them to be added together. But the number of trainable weights are significantly smaller since the Rank ($r$) is small.

Usually the trainable weights are kept to be `~1%` of the total parameters.

### Initialization
When training starts, we want the model to behave exactly like the pre-trained model (so we don't break it immediately).
- **Matrix $A$** is initialized with **random Gaussian initialization**.
- **Matrix $B$** is initialized to **zeros**.
Therefore, at step 0:

$$BAx = 0 \cdot A \cdot x = 0$$

$$h = W_0x + 0$$
i.e. The model starts exactly at the pre-trained baseline.

### The Scaling Factor ($\alpha$)
While implementing (like `peft`), you will see a hyperparameter `alpha`. The actual equation used is:

$$h = W_0x + \frac{\alpha}{r} (BAx)$$
- **$\alpha$ (Alpha):** A scaling constant.

**Why do we divide by $r$?**

For **Hyperparameter Stability**.
It acts as a normalizer. If you decide to try a different Rank $r$ later (e.g., switching from $r=8$ to $r=16$), you don't want to have to re-tune your learning rate entirely. The fraction $\frac{\alpha}{r}$ ensures that the magnitude of the update remains roughly consistent even if you change the rank.
The scaling factor $\frac{\alpha}{r}$ exists so that **you don't have to manually retune Learning Rate (LR) every time you try a different Rank.**

#### More Details

Let's look at what happens inside the matrix multiplication $W_{new} = B \times A$.

For a specific element in the output matrix (let's call it $y$), the calculation is a **dot product** involving the rank dimension.

$$y = \sum_{i=1}^{r} B_{i} \cdot A_{i}$$

- **If $r = 8$:** You are summing 8 numbers.
- **If $r = 128$:** You are summing 128 numbers.
If the values inside $A$ and $B$ are initialized roughly the same way (e.g., using a standard Gaussian distribution), **summing 128 numbers results in a much larger total value than summing 8 numbers.**

Imagine you are an AI engineer. You spend a week finding the perfect Learning Rate (let's say $0.0003$) for a Rank of **8**.

Now, you decide: _"I want the model to be smarter, so I will increase the Rank to **64** and retrain."_

**Without the $\frac{1}{r}$ division:**

1. The matrix multiplication $BA$ would naturally output values roughly **8x larger** (because $64 / 8 = 8$) simply because there are more terms in the summation.
2. This means the update $\Delta W$ is suddenly huge.
3. Your gradients will explode.
4. Your previously perfect Learning Rate ($0.0003$) is now way too high. The model diverges or crashes.
5. You have to waste days finding a _new_ Learning Rate for $r=64$.


By dividing the output of $BA$ by $r$, we neutralize the effect of the summation length.

$$Update = \frac{\alpha}{r} (B \times A)$$

- If $r$ increases, the raw sum $(B \times A)$ gets bigger.
- But the denominator $r$ also gets bigger.
- These two effects cancel each other out.

**The Result:** The "signal strength" of the update remains roughly constant regardless of the Rank. This allows you to keep the same Learning Rate you used for $r=8$ and apply it to $r=64$, $r=128$, etc., with a high probability that it will train stably.


**The Role of Alpha ($\alpha$)**
If $r$ cancels out the math changes, what is $\alpha$ for?
Think of $\alpha$ as a **"Volume Knob"** for the LoRA adapter.
- **$W_0$** is the existing knowledge (the base model).
- **$\Delta W$** is the new knowledge (the LoRA adapter).

You want to balance how much the model listens to the old knowledge vs. the new knowledge.

In the equation $h = W_0x + \frac{\alpha}{r} (BAx)$:
- If you set $\alpha = r$, the scaling factor is 1. The adapter behaves "normally."
- If you set $\alpha = 2r$, the scaling factor is 2. You are mathematically doubling the influence of the LoRA weights relative to the base weights.


### How to choose the Rank?
The Rank determines the **capacity** of your fine-tuning. It controls how "complex" the new behaviors can be.
- **Low Rank ($r = 8$ or $16$):**
    - **Use Case:** Style transfer, formatting, changing the "voice" of the model (e.g., making it sound like a pirate), or simple classification tasks.
    - **Why:** The original LoRA paper showed that for many tasks, the "intrinsic dimension" of the change is extremely low. You don't need a massive matrix to learn to put a JSON bracket at the end of a sentence.
    - **Benefit:** Fastest training, lowest VRAM usage.

- **High Rank ($r = 64, 128, 256$):**
    - **Use Case:** Complex reasoning, teaching the model _new_ knowledge it didn't have before, or heavy logic tasks (e.g., coding, math).
    - **Why:** If you are trying to teach the model details about a specific biology textbook it has never seen, the update is not "low rank", it involves complex new connections. You need a larger matrix to store this information.
    - **Note:** There are diminishing returns. Going from $r=8$ to $r=64$ often boosts performance. Going from $r=64$ to $r=256$ often yields barely any improvement but triples the training cost.


### How to choose Alpha ($\alpha$)

Empirically, using a higher Alpha often stabilizes training for LoRA. It acts like a momentum booster.
**($\alpha = 2r$)

It is critical to understand that **Alpha and Learning Rate (LR) are mathematically coupled.**
If you double your Alpha, you are mathematically doubling the magnitude of your update. This is roughly equivalent to doubling your Learning Rate.

Therefore, if you find a configuration that works:
- $r=16, \alpha=16, LR=2e^{-4}$

And you decide to change Alpha to 32:
- $r=16, \alpha=32$ ... you should probably halve your LR to $1e^{-4}$ to keep the training dynamics similar.

Usually **fix Alpha** (e.g., always at 16 or 32) and then strictly tune the Learning Rate. Do not try to tune both simultaneously; you will chase your tail.
### Merge for Inference
One of the best features of LoRA is that it introduces **zero latency** during inference (production).
Because $BA$ has the same dimensions as $W_0$, once you are done training, you can simply perform matrix addition to permanently fuse the weights:

$$W_{final} = W_0 + (B \times A) \cdot \frac{\alpha}{r}$$
You can now discard $A$ and $B$ and serve $W_{final}$ as a standard model.


## Target Modules
A modern LLM (like Llama-3) is just a stack of identical "blocks" (usually 32 or 80 of them). Inside _each_ block, there are two distinct departments that perform different jobs:
1. The Attention Mechanism
2. The Feed-Forward Network / MLP

We apply LoRA to the linear layers (matrices) inside these departments.

### Department A: The Attention Mechanism (Q, K, V, O)
Imagine the token "Bank" is trying to figure out if it means "River Bank" or "Financial Bank." It needs to look at the other words in the sentence (context). It does this using four specific projection layers.
- **`q_proj` (Query):** "What am I looking for?"
    - The token broadcasts a search query. (e.g., "I am 'Bank', looking for words like 'money' or 'water' nearby.")

- **`k_proj` (Key):** "What do I contain?"
    - Every other token holds up a label. (e.g., The word "River" holds up a label saying "Nature/Water context".)

- **`v_proj` (Value):** "What information do I pass along?"
    - If the Query matches the Key, this is the actual information transferred. (e.g., "River" passes the concept of "Water" to "Bank".)

- **`o_proj` (Output):** "Mix it all together."
    - After grabbing information from all relevant words, this layer blends it back into the token's current state.

**Why target these?** In the early days of LoRA, people _only_ targeted `q_proj` and `v_proj`. This was enough to change "what the model focuses on," but it wasn't great at teaching the model new logic.

---

### Department B: The MLP / Feed-Forward (Gate, Up, Down)
After the Attention step, the token "Bank" knows it refers to a river. Now it needs to _think_ about that. It enters the MLP (Multi-Layer Perceptron). In Llama-style models, this is a "SwiGLU" architecture, which uses three specific matrices:
- **`up_proj` (Up Projection):**
    - Takes the input (e.g., dimension 4096) and **explodes** it into a much higher dimension (e.g., 14,336). This is where the model "unpacks" the concept to look at fine-grained details.

- **`gate_proj` (Gate Projection):**
    - This works in parallel with the Up projection. It acts as a filter (using the SiLU activation function). It decides which parts of the "exploded" information are important and which should be ignored.

- **`down_proj` (Down Projection):**
    - Takes the filtered, high-dimensional data and **compresses** it back down to the original size (4096) so it can be passed to the next block.

**Why target these?** These layers represent the model's "knowledge repository" and reasoning circuits. If you want to teach the model **complex math**, **coding**, or **a new language**, you _must_ target these MLP layers.

### Embedding / Unembedding head
We usually **do not** apply LoRA to the `lm_head` (the very final layer that predicts the next word) or the `embed_tokens` (the very first layer), unless you are adding new tokens to the vocabulary (like training a model on a new language with new characters).
## Resources

---
**Back to**: [[ML & AI Index]]
