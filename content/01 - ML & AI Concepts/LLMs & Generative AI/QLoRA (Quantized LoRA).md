
# Overview
**QLoRA (Quantized Low-Rank Adaptation)** is the technique that democratized LLM fine-tuning. Before QLoRA (May 2023), fine-tuning a 70B model required an industrial server cluster. With QLoRA, you can do it on a dual-GPU workstation or even a high-end gaming PC.

QLoRA tackles the biggest memory hog: the frozen base model. It compresses the base model from **16-bit** down to **4-bit**. reducing the memory taken by 75%
This is called **4-bit Quantization**.

If it's so simple, why didn't we just do this before?
The answer is that standard 4-bit quantization destroys model performance. If you dumb down the model too much, it becomes "brain damaged" and can't facilitate the fine-tuning process.

# Key Innovations to make QLoRA work
### 4-bit NormalFloat (NF4)
Standard quantization (Integer Quantization) assumes data is spread out evenly (Uniform distribution).
- It chops the range of numbers into equal-sized buckets.

However, Neural Network weights generally follow a **Normal (Gaussian) Distribution** (Bell Curve). Most weights are clustered near zero; very few are outliers.

**NF4** is a data type designed mathematically to fit the shape of neural network weights.
- Allocates **more buckets** near zero (where most information lives).
- Allocates **fewer buckets** at the extremes.

**Result:** NF4 captures the "resolution" of the model significantly better than standard 4-bit Integers, resulting in almost zero performance degradation compared to 16-bit.

![[QLoRA (Quantized LoRA) 2025-12-30 22.35.06.excalidraw.svg]]


### Double Quantization
To decompress the 4-bit weights back to 16-bit for math, the model needs "Quantization Constants" (scaling factors).
- Usually, we have one constant for every block of 64 parameters.
- These constants _themselves_ are 32-bit numbers.
- Across billions of parameters, these constants add up to significant VRAM (hundreds of MBs).

**Double Quantization** creates a second layer: it **quantizes the quantization constants**.
- It compresses the 32-bit constants into 8-bit.

**Result:** This saves an extra ~0.5 GB of VRAM on a 65B model without hurting performance significantly.

#### Quantization
##### How to squeeze 64 16-bit numbers into 4-bit integers using single constant? (Simplified Example)
Imagine you have a list of precise decimal numbers (32-bit Floats). These are your weights.
You want to store them as **4-bit Integers**.
- A 4-bit integer can only hold whole numbers from **-8 to +7**.
- You cannot fit `2.5` into a box that only accepts integers.

We need to "shrink" the real numbers so they fit into the integer box.

Let's look at a simplified "Block" of 3 weights.
[0.1, 2.5, -1.5]

**Step A: Find the Absolute Max**
Look at the block. What is the number with the largest magnitude? It is 2.5.
This number defines the "Range" of this block.

**Step B: Calculate the Constant (Scaling Factor)**
We need to map our largest number (2.5) to the largest container slot available (7).

$$Constant = \frac{\text{Absolute Max of Block}}{\text{Max Integer Container}}$$

$$Constant = \frac{2.5}{7} \approx 0.357$$
**This `0.357` is your Quantization Constant.** It is the "key" to this specific block.

**Step C: Quantize (Divide and Round)**
Now we divide every number in the block by the constant and round to the nearest integer.
1. **0.1:** $0.1 / 0.357 = 0.28 \rightarrow \text{Round to } \mathbf{0}$
2. **2.5:** $2.5 / 0.357 = 7.00 \rightarrow \text{Round to } \mathbf{7}$
3. **-1.5:** $-1.5 / 0.357 = -4.2 \rightarrow \text{Round to } \mathbf{-4}$

**What we store in memory:**
- The Weights: `[0, 7, -4]` (These are tiny 4-bit integers).
- The Constant: `0.357` (This is a 32-bit float).
##### De-Quantization (Reading the data)
When the model needs to use these weights for math during training/inference, it reverses the process.
$$Real \approx Integer \times Constant$$

1. $0 \times 0.357 = \mathbf{0}$ (Original was 0.1, we lost some precision)
2. $7 \times 0.357 = \mathbf{2.499}$ (Original was 2.5, very close!)
3. $-4 \times 0.357 = \mathbf{-1.428}$ (Original was -1.5, close!)
##### Why do we use Blocks?
Why not one Constant for the whole model?
This is the "Outlier" problem.

Imagine you have a block of 64 numbers.
- 63 of them are small (e.g., `0.1`, `0.2`).
- **1 of them is huge** (e.g., `100.0`).

If you calculate the Constant based on the 100.0:

$$Constant = 100 / 7 = 14.2$$
Now try to quantize the small number 0.1:

$$0.1 / 14.2 = 0.007 \rightarrow \text{Round to } \mathbf{0}$$
The outlier destroyed the precision of the small numbers. All your detailed small weights became zeros.
By breaking the model into small blocks (64 weights), we isolate the outliers. If one block has a huge number, only that block gets a large Constant. The neighbor block might have small numbers and a small Constant, preserving its precision.



In standard integer quantization (explained above), no. It's just simple division and multiplication.


##### NF4 in QLoRA
In QLoRA specifically, they use NF4 (NormalFloat 4).
NF4 is effectively a mathematically generated Lookup Table.
- Since we know neural weights look like a Bell Curve, NF4 says: "Don't space the integers -8 to 7 evenly."
- Instead, let's put more integer slots near 0 (where most data is) and fewer slots at the edges.
- The **Quantization Constant** is still used to stretch or shrink that Bell Curve to fit the specific block's magnitude.

### Paged Optimizers
This is a failsafe feature. Occasionally during training, you get a "gradient spike", a complex batch causes memory usage to explode momentarily. Usually, this crashes your run (OOM Error).

QLoRA utilizes **Unified Memory** (a feature of NVIDIA GPUs).
- If VRAM fills up, it automatically "pages" the Optimizer States (the training memory) to your computer's **System RAM (CPU RAM)**.
- It's slower, but it prevents the crash. Once the spike passes, it moves data back to VRAM.

#### What causes CUDA OOM?
It is rarely the model itself that kills you, it is the **Optimizer**.

When you train a model, you are not just storing the weights ($W$). You are also storing the "notes" the optimizer keeps to decide how to update those weights.

Most LLM training uses the **AdamW** optimizer. AdamW is smart, but memory-hungry. For _every single parameter_ in your model, AdamW maintains two additional statistics:

1. **First Moment ($m$):** The moving average of the gradient (Momentum).
2. **Second Moment ($v$):** The moving average of the squared gradient (Variance).

**The Math of the Memory Footprint:**
- **Model Weight:** 4-bit (0.5 bytes).
- **Gradient:** Bfloat16 (2 bytes).
- **Optimizer State 1 ($m$):** FP32 (4 bytes) ,  _High precision required here._
- **Optimizer State 2 ($v$):** FP32 (4 bytes) , _High precision required here._

The "metadata" for the optimizer (8 bytes per parameter) is 16 times larger than the model weight itself (0.5 bytes per parameter)!

#### Why The  Memory Spike?
Memory usage during training is not a flat line.
1. **Forward Pass:** As layers calculate, we store "Activations" (intermediate results) so we can calculate gradients later. **Memory Usage Spikes Up.**
2. **Backward Pass:** We use the activations to calculate gradients. **Memory Usage Peaks.**
3. **Optimizer Step:** We update the weights using the Optimizer States.

If the peak of that rollercoaster goes even 1MB over your GPU's limit, the run crashes.

#### Why it works well for QLoRA?
Moving data back and forth takes time. 
But since QLoRA is usually compute-bound (the math takes a long time), the slight delay of copying data over PCIe is often masked by the computation time. You barely notice the slowdown, but you gain the ability to fine-tune significantly larger batches or models.

# Computation Flow
This is the most critical concept to grasp: **The computation is NOT done in 4-bit.**
You cannot perform stable Gradient Descent in 4-bit. It's too jagged.

QLoRA uses a "De-quantize on the fly" mechanism.

**The Loop:**
1. **Storage:** The Base Model ($W_0$) sits in VRAM in **4-bit NF4** (Tiny).
2. **Forward Pass:** As data flows through a layer:
    - The specific block of weights needed is retrieved.
    - It is instantly **de-quantized** to **Bfloat16**.
    - The Matrix Multiplication happens in highly precise **Bfloat16**.
    - The result is passed on.
    - The de-quantized Bfloat16 weights are discarded (clearing memory).
3. **Adapter ($A, B$):** The LoRA adapters are always kept in **Bfloat16** or **Float32**. They are never quantized.


$$Y = (\text{dequant}(W_{4bit}) + BA) X$$

This hybrid approach gives you the **memory footprint of 4-bit** storage but the **mathematical precision of 16-bit** computation.


---

**Progress**: 
- [x] Read overview materials
- [x] Understand key concepts
- [x] Review mathematical foundations
- [ ] Study implementations
- [ ] Complete hands-on practice
- [x] Can explain to others

**Status Options**: `not-started` | `in-progress` | `completed` | `review-needed`
**Difficulty Options**: `beginner` | `intermediate` | `advanced` | `expert`

---
**Back to**: [[ML & AI Index]]
