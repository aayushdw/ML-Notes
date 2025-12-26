## Overview
Fine-tuning is effectively the bridge between a "generic knowledgeable" model (like a base GPT or Llama model) and a "specialized expert" model (like a medical coding assistant or a brand-specific customer support agent).

## LLM Lifecycle
1. **Pre-Training (Unsupervised):** The model reads trillions of tokens (internet data) to learn probability distributions. It learns that after "The capital of France is," the next word is likely "Paris."
2. **SFT (Supervised):** We feed the model specific `{Instruction, Response}` pairs. We tell it: "When a user asks X, you should reply with Y."
3. **RLHF/DPO (Preference Alignment):** (Optional) We further refine the model based on human preferences (ranking which answer is better).

## How SFT is Achieved
Technically, SFT uses the same underlying mechanism as pre-training—**Next Token Prediction**—but with a crucial twist regarding the data and the loss function.
### The Dataset Structure
In pre-training, data is raw text. In SFT, data is structured. A typical SFT entry looks like:
- **Prompt:** "Explain quantum entanglement like I'm five."
- **Completion:** "Imagine you have two magic dice..."

### Data Formatting (Templating)
The model doesn't understand "User" or "AI" naturally. We must wrap the data in a specific **Prompt Template**. If you mess up the template, the model's performance collapses.

_Example (ChatML format):_
```
<|im_start|>system
You are a helpful physics teacher.<|im_end|>
<|im_start|>user
Explain quantum entanglement.<|im_end|>
<|im_start|>assistant
Imagine two magic dice...<|im_end|>
```

### The Loss Masking (Crucial Step)
This is the most technical differentiator. When training, we feed the whole sequence into the model. However, we **do not** want the model to learn how to predict the user's prompt; we only want it to improve at generating the _assistant's response_.

To achieve this, we use a **Loss Mask**.
- **User Tokens:** Loss is set to 0 (ignored). Backpropagation does not update weights based on these tokens.
- **Assistant Tokens:** Loss is calculated (Cross-Entropy). The model is penalized if it fails to predict the next token in the "Assistant" section.

## Fine Tuning Architecture

### Full Fine-Tuning
You load the entire model into memory and update _every single parameter_ (weight) based on the SFT dataset.
- **Pros:** Potentially higher theoretical performance ceiling for massive domain shifts.
- **Cons:** Extremely expensive. To fine-tune a 70B parameter model, you might need 4-8 H100 GPUs with hundreds of gigabytes of VRAM just to store the optimizer states.
- **Risk:** Catastrophic Forgetting (the model learns the new task but forgets general English).


### PEFT (Parameter-Efficient Fine-Tuning)
Industry standard for 95% of use cases. Instead of updating all weights, we freeze the base model and only train a tiny subset of new parameters.

#### [[LoRA (Low-Rank Adaptation)]]
Most popular PEFT technique.
- **Concept:** Instead of updating a massive weight matrix $W$, LoRA injects two small matrices, $A$ and $B$, next to it.
- The new output is $h = W_0x + BAx$.
    - $W_0$ is frozen (the original model).
    - $A$ and $B$ are trainable "adapter" matrices.

- This way you can fine-tune a 70B model on a single consumer GPU because you are only training <1% of the parameters.

#### [[QLoRA (Quantized LoRA)]]
This takes LoRA a step further by compressing the base model.
- The frozen base model ($W_0$) is loaded in **4-bit precision** (drastically reducing memory usage).
- The LoRA adapters ($A$ and $B$) are kept in higher precision (16-bit or 32-bit) for accurate training.
- **Impact:** This allows you to fine-tune massive models (like Llama-3-70b) on relatively cheap hardware.

#### Generic Adapter Method
Both LoRA and Adapter methods rely on the exact same Low-Rank Matrix Decomposition trick.
They both squash a large vector into a small vector ($d \to r$) and then expand it back ($r \to d$).
However, the **Topological Placement** and the **Non-Linearity** make them fundamentally different in engineering practice.

##### The Critical Difference: Non-Linearity
This is the most important distinction that prevents standard Adapters from being merged.
- Bottleneck Adapter: Has a non-linear activation function (ReLU, GELU) between the down and up projections.

$$h_{out} = h_{in} + W_{up} \cdot \mathbf{ReLU}(W_{down} \cdot h_{in})$$

- LoRA: Is purely linear. There is no activation function between the two matrices.

$$h_{out} = h_{in} + W_{up} \cdot W_{down} \cdot h_{in}$$


Because LoRA is purely linear, $W_{up} \cdot W_{down}$ collapses into a single matrix $\Delta W$. This allows you to add it directly to the frozen weights: $W_{frozen} + \Delta W$.
You cannot do this with Adapters because the ReLU makes the operation non-linear. You cannot merge $W_{up} \cdot \text{ReLU}(W_{down})$ into a single linear matrix.

#### Soft Prompting / P-Tuning (TODO)
Instead of changing weights inside the model, you train "virtual tokens" (embeddings) that are prepended to the input. It's like finding the perfect magic words to whisper to the model to get it to behave, without performing surgery on the model's brain.
[[P-Tuning, Prefix-Tuning]]


## SFT Training Data Preparation
Most common point of failure in fine-tuning. You can have the perfect Rank, Alpha, and Learning Rate, but if your data formatting is slightly off, the model will learn garbage.
In SFT, we are not just feeding text to the model; we are feeding it **Logic Gates** disguised as text.

### Tricking the Loss Calculation
In Pre-training, the model tries to predict _every_ token. In SFT, we want the model to predict **only the Assistant's response**.
If we don't mask the User's prompt, the model will mistakenly learn to "imitate the user" rather than "answer the user."

**The Data Tensors:** When you feed a batch to the GPU, you are actually feeding two identical tensors:
1. **`input_ids`**: The tokenized numbers of the full conversation.
2. **`labels`**: The target numbers the model should predict.

**The Trick:** We take the `labels` tensor and replace the User's tokens with a special number: **-100**. In PyTorch, `-100` is the universal "Ignore Index." When the Cross-Entropy Loss encounters `-100`, it skips that token entirely. No gradients are calculated.

### JSONL Container
The industry standard file format is **JSONL** (JSON Lines). Every line in the file is a separate, valid JSON object representing one full conversation.
Raw Structure:
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

Note that model cannot read JSON objects. It can only read a **single string of text**. We need a **Template** to flatten this JSON into a string.


### Prompt Templates
#### ChatML (Modern Standard)
Introduced by OpenAI/Microsoft, adopted by the open-source community (e.g., Qwen, Mistral). It uses explicit "Special Tokens" to delineate turns.

```text
<|im_start|>system
You are a helpful AI.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

- `<|im_start|>`  is a single unique token ID added to the vocabulary. It acts as a hard separator that the model learns to recognize as "New Speaker."

#### Llama-3 Format
Llama-3 uses a very specific, verbose set of header tags.
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_response}<|eot_id|>
```


### Visualizing the Full Tokenization Stream
**The Conversation:** User: "Hi" Assistant: "Ho"

**Step 1: Flatten to String (using Template)** `"<|user|>Hi<|end|><|assistant|>Ho<|end|>"`

**Step 2: Tokenize (Turn into IDs)** `[128, 55, 129, 44, 55, 67]`

**Step 3: Create Labels (Copy IDs)** `[128, 55, 129, 44, 55, 67]`

**Step 4: Apply The Mask (The Critical Step)** We find the indices that correspond to the User's turn and the System tokens. We overwrite them with **-100**.
- **Input IDs:** `[128, 55, 129, 44, 55, 67]` (Model sees everything)
- **Labels:** `[-100, -100, -100, 44, 55, 67]` (Model is only graded on "Ho<|end|>")


## Important Considerations

### The "Instruction Following" Tax
SFT aligns the model to instructions, but it often reduces the diversity of the output. A base model might write a creative, rambling story. An SFT model is trained to be concise and helpful. If you over-train (too many epochs), the model becomes repetitive and "robotic."
#### Dataset Quality > Quantity
In Pre-training, quantity is important. In SFT, quality is important.
- **LIMA Hypothesis:** A paper titled "LIMA: Less Is More for Alignment" showed that a model fine-tuned on only 1,000 incredibly high-quality, hand-curated examples could beat models trained on 50,000 mediocre examples.

## Resources


---
**Back to**: [[ML & AI Index]]
