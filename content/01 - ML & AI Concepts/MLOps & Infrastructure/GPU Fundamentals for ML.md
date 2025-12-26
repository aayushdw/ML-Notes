# GPU Fundamentals for ML

## Overview
Graphics Processing Units (GPUs) are the engine of modern Deep Learning. Unlike CPUs, which are designed for complex logic and sequential processing, GPUs are specialized for massive parallelism. In the context of ML, they are essentially **high-throughput matrix multiplication engines**.

Understanding GPU architecture is critical for **MLOps** because it dictates how we optimize training speed, reduce inference latency, and debug performance bottlenecks (computational vs. memory).

## Key Ideas / Intuition

### The Analogy: Ferrari vs. Cargo Ship
*   **CPU (The Ferrari)**: Designed to move a small amount of "cargo" (instructions) extremely fast. It has huge caches and complex control logic to minimize latency for every single instruction. Great for sequential tasks like OS management or running a web server.
*   **GPU (The Cargo Ship)**: Designed to move massive amounts of "cargo" (data) at once. It takes longer to get started (latency), but once it's moving, it carries thousands of containers simultaneously. Great for tasks where the same operation is applied to millions of data points (e.g., pixel rendering, matrix multiplication).

### SIMT (Single Instruction, Multiple Threads)
This is the core execution model of NVIDIA GPUs.
*   **Concept**: You write one piece of code (a **Kernel**), and the GPU spawns thousands of threads to execute that same code on different parts of data simultaneously.

### Kernel Launch
*   **Host**: The CPU. It orchestrates the program, loads data, and tells the GPU what to do.
*   **Device**: The GPU. It executes the heavy computational kernels.
*   **Asynchronous Nature**: The CPU launches a kernel and immediately moves on. It doesn't wait for the GPU to finish unless explicitly told to (`cudaDeviceSynchronize()`).

## Hardware Architecture

To optimize ML models, you must understand the physical hierarchy of the GPU.

### 1. Compute Units
*   **SM (Streaming Multiprocessor)**: The fundamental building block. An A100 GPU has 108 SMs. Each SM contains:
    *   **CUDA Cores**: Specialized for FP32, FP64, and INT32 operations.
    *   **Tensor Cores**: Specialized hardware for Matrix Multiplication ($D = A \times B + C$). They perform mixed-precision math (FP16/BF16) much faster than standard CUDA cores.
    *   **L1 Cache / Shared Memory**: Ultra-fast memory shared by threads within the same block.

### 2. Memory Hierarchy (The Speed vs. Size Trade-off)
Data movement is often the bottleneck, not compute.

| Memory Type     | Location          | Size (A100) | Bandwidth     | Latency     |
| :-------------- | :---------------- | :---------- | :------------ | :---------- |
| **Registers**   | On-chip (closest) | ~256KB/SM   | ~100s TB/s    | ~0 cycles   |
| **L1 / Shared** | On-chip           | ~192KB/SM   | ~19 TB/s      | ~20 cycles  |
| **L2 Cache**    | On-chip (shared)  | 40-80 MB    | ~5 TB/s       | ~200 cycles |
| **HBM (VRAM)**  | Off-chip (Device) | 40-80 GB    | ~1.5 - 2 TB/s | ~400 cycles |
| **System RAM**  | Host (CPU)        | >100 GB     | ~50-100 GB/s  | Very slow   |

> [!IMPORTANT]
> **Performance Tip**: fast GPU code keeps data in Registers and L1 Cache as much as possible. Fetching from HBM (VRAM) is expensive. Fetching from System RAM (over PCIe) is catastrophic for performance.

## Mathematical Foundation

### Arithmetic Intensity
This metric determines if your workload is **Compute Bound** or **Memory Bound**.

$$ \text{Arithmetic Intensity} = \frac{\text{Total FLOPs (Floating Point Operations)}}{\text{Total Bytes Accessed}} $$

*   **Compute Bound**: You spend most time doing math (High Intensity). Faster Cores help.
    *   *Example*: ResNet-50 Conv Layers, Large Matrix Multiplications.
*   **Memory Bound**: You spend most time moving data (Low Intensity). Faster Memory helps.
    *   *Example*: LayerNorm, Activation functions (ReLU), Element-wise additions.

### The Roofline Model
A visual representation of performance limits.
*   **Slanted Ceiling**: Limited by Memory Bandwidth (low arithmetic intensity).
*   **Flat Ceiling**: Limited by Peak Compute capability (high arithmetic intensity).
*   **Goal**: Move your kernel "up and to the right" (increase intensity to utilize peak compute).

### Tensor Core Math
Tensor Cores accelerate the operation:
$$ D = A \times B + C $$
Where $A$ and $B$ are $4 \times 4$ matrices (in FP16) and $C, D$ are accumulation matrices (in FP32).
*   **Benefit**: Performs roughly $64$ (or more on newer gens) FMA (Fused Multiply-Add) operations per clock cycle, vs $1$ FMA on a standard core.

## Practical Application (MLOps)

### 1. GPU Utilization (The Lie)
Running `nvidia-smi` and seeing `100%` utilization does **not** mean your code is efficient.
*   **What it means**: "At least one warp was active on the GPU during the sample period."
*   **Reality**: You could be using 1% of the compute power but stalling on memory 99% of the time, and `nvidia-smi` will still show 100%.
*   **Use Profilers**: Use Nsight Systems or PyTorch Profiler to see *SM Efficiency* and *Memory Bandwidth Utilization*.

### 2. Bottlenecks
*   **Data Loader Bottleneck**: If GPU utilization drops to 0% periodically, your CPU cannot feed data fast enough.
    *   *Fix*: Increase `num_workers`, use pre-fetching.
*   **Memory Bandwidth Bottleneck**: High utilization but low throughput.
    *   *Fix*: Fused kernels (combining operations to read data once), Quantization (move less data).

### 3. Precision & Quantization
Using lower precision reduces memory traffic and increases tensor core throughput.
*   **FP32 (Single)**: 4 bytes. The default for safety.
*   **TF32 (Tensor Float)**: 19 bits. Nvidia's magic format on Ampere+. Works like FP32 but runs on Tensor Cores.
*   **FP16 / BF16**: 2 bytes. **Standard for LLM training**. BF16 is preferred as it preserves the dynamic range of FP32.
*   **INT8**: 1 byte. Used for *Inference* (Quantization).

## Comparisons

| Feature | CPU | GPU | TPU (Google) |
| :--- | :--- | :--- | :--- |
| **Core Count** | Few (tens) powerful cores | Many (thousands) simple cores | Massive Systolic Array |
| **Design Goal** | Low Latency, Complex Logic | High Throughput, Parallelism | Matrix Mult. specialization |
| **Memory** | DDR (Low bandwidth, huge capacity) | HBM (High bandwidth, limited capacity) | HBM |
| **Flexibility** | High (Any C++ code) | Medium (CUDA/Kernels) | Low (XLA Graph operations) |

### Consumer vs. Datacenter
*   **Consumer (RTX 4090)**: Great FP32 performance, but no NVLink (cannot effectively pool memory across cards for training huge models), no ECC (error correction).
*   **Datacenter (A100/H100)**: Massive HBM capacity (80GB), NVLink (600 GB/s interconnect), ECC, Multi-Instance GPU (MIG).

## Resources
*   [Tim Dettmers: Which GPU to buy for Deep Learning](https://timdettmers.com/) - The bible of GPU hardware analysis.
*   [Nvidia Blog: Tensor Cores](https://developer.nvidia.com/tensor-cores)
*   [Horace He: Making Deep Learning Go Brrr From First Principles](https://horace.io/brrr_intro.html) - excellent deeper dive.

## Progress Checklist
- [ ] Read overview
- [ ] Understand key concepts (SIMT, Latency vs Throughput)
- [ ] Review math (Arithmetic Intensity)
- [ ] Hands-on practice (Profile a training run with PyTorch Profiler)
- [ ] Can explain to others

**Back to**: [[03 - MLOps & Infrastructure Index]]
