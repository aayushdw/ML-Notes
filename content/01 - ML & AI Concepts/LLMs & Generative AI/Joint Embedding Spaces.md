## Overview

Joint embedding spaces are learned representations where data from different modalities (text, images, audio, video) are mapped into a shared vector space. The fundamental insight is that semantically similar concepts across modalities should be close together in this space, regardless of their original format. A picture of a dog and the phrase "a golden retriever playing fetch" should have similar vector representations.

This approach enables **cross-modal reasoning**: comparing, retrieving, and generating content across modalities without explicit paired supervision for every possible combination. Joint embeddings are the foundation of modern multimodal AI systems like CLIP, GPT-4V, Gemini, and LLaVA.

## Key Ideas & Intuition

### The Core Problem: Modality Gap

Each data modality has fundamentally different statistical properties:
- **Images**: Dense pixel grids, spatial relationships, local patterns
- **Text**: Sequential tokens, discrete symbols, compositional semantics
- **Audio**: Temporal waveforms, frequency spectra

Traditional unimodal models encode each type into incompatible vector spaces. A ResNet image embedding and a BERT text embedding cannot be directly compared because they were trained on different objectives with different architectures.

```
Before Joint Embeddings:
┌─────────────────┐     ┌─────────────────┐
│  Image Encoder  │     │  Text Encoder   │
│    (ResNet)     │     │    (BERT)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    [512-dim]               [768-dim]
    Image Space             Text Space
         │                       │
         └──────── ✗ ────────────┘
              Incompatible!
```

### Solution: Shared Representation Space

Joint embedding methods train encoders for each modality to project into a **common vector space** where semantic similarity translates to geometric proximity (typically measured by cosine similarity or Euclidean distance).

```
After Joint Embedding Training:
┌─────────────────┐     ┌─────────────────┐
│  Image Encoder  │     │  Text Encoder   │
│   (ViT/ResNet)  │     │ (Transformer)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    Projection               Projection
       Head                     Head
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Shared Space   │
            │   (e.g., 512d)  │
            │                 │
            │  "dog" ≈ 🐕     │
            │  "cat" ≈ 🐱     │
            └─────────────────┘
```

### Why This Works: Semantic Anchoring

The key insight is that **language provides semantic anchors**. Humans have already organized concepts into linguistic categories. By aligning visual (or audio) representations to these language-based anchors, we inherit the compositional structure of language.

For example, if the model learns:
- "dog" → region A in the space
- "golden" → modifies toward region B
- "playing" → activates region C

Then "golden retriever playing" naturally composes these learned directions, and images matching this description cluster in the same region without ever seeing that exact phrase during training.

---

## Architectures for Multimodal Learning

### 1. Dual-Encoder Architecture (Contrastive)

The most influential approach, pioneered by **CLIP** and **ALIGN**.

```
           Image                    Text
             │                        │
             ▼                        ▼
    ┌────────────────┐      ┌────────────────┐
    │ Vision Encoder │      │  Text Encoder  │
    │   (ViT-L/14)   │      │  (Transformer) │
    └───────┬────────┘      └───────┬────────┘
            │                       │
            ▼                       ▼
    ┌────────────────┐      ┌────────────────┐
    │ Linear Project │      │ Linear Project │
    │   (768→512)    │      │   (512→512)    │
    └───────┬────────┘      └───────┬────────┘
            │                       │
            ▼                       ▼
         z_img                   z_text
            │                       │
            └─────────┬─────────────┘
                      ▼
              Cosine Similarity
              ───────────────
              Contrastive Loss
```

**Key Properties:**
- Encoders are **independent**: Can compute image or text embeddings separately
- Enables **efficient retrieval**: Pre-compute all image embeddings, then query with text
- **Zero-shot transfer**: New categories described in text can be matched without retraining

### 2. Cross-Attention / Fusion Architecture

Used in models like **Flamingo**, **BLIP-2**, and **LLaVA** for deeper multimodal reasoning.

```
┌────────────────────────────────────────────────────────┐
│                    Text Tokens                         │
│           [CLS] A dog playing in the park [SEP]        │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Self-Attention    │
              │   (Text Pathway)    │
              └─────────┬───────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐   ┌───────────┐   ┌─────────┐
    │ Query   │   │   Key     │   │  Value  │
    │ (Text)  │   │ (Image)   │   │ (Image) │
    └────┬────┘   └─────┬─────┘   └────┬────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
              ┌─────────────────────┐
              │  Cross-Attention    │◄── Image Patches
              │  (Fuse Modalities)  │    from ViT
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Feed Forward      │
              │   + More Layers     │
              └─────────────────────┘
```

**Key Properties:**
- Enables **fine-grained reasoning** about specific image regions
- Text can **attend to visual details**: "What color is the leftmost object?"
- Higher computational cost than dual-encoder

### 3. Decoder-Only Multimodal (Modern VLMs)

Models like **GPT-4V**, **Gemini**, and **Claude** use a different paradigm:

```
         Image                    Text Prompt
           │                          │
           ▼                          │
   ┌───────────────┐                  │
   │ Vision Encoder│                  │
   │    (ViT)      │                  │
   └───────┬───────┘                  │
           │                          │
           ▼                          │
   ┌───────────────┐                  │
   │  Adapter /    │                  │
   │  Projector    │                  │
   │ (MLP/Q-Former)│                  │
   └───────┬───────┘                  │
           │                          │
           └──────────────────────────┘
                       │
                       ▼
           ┌───────────────────┐
           │  Visual tokens +  │
           │  Text tokens      │
           │  [img][img]...[txt]│
           └─────────┬─────────┘
                     │
                     ▼
           ┌───────────────────┐
           │  Autoregressive   │
           │     LLM Core      │
           │  (Decoder-only)   │
           └─────────┬─────────┘
                     │
                     ▼
                  Output
```

**Key Insight**: Visual information is converted into "visual tokens" that the LLM treats like text tokens. The LLM's pretraining knowledge about language and reasoning transfers to multimodal tasks.

---

## Mathematical Foundation

### Contrastive Learning Objective (InfoNCE/CLIP Loss)

Given a batch of $N$ image-text pairs $(x_i, t_i)$, the goal is to maximize similarity for matched pairs and minimize it for mismatched pairs.

Let $\mathbf{z}_i^{\text{img}} = f_{\theta}(x_i)$ be the normalized image embedding and $\mathbf{z}_i^{\text{txt}} = g_{\phi}(t_i)$ be the normalized text embedding.

**Similarity Matrix:**
$$S_{ij} = \frac{\mathbf{z}_i^{\text{img}} \cdot \mathbf{z}_j^{\text{txt}}}{\tau}$$

where $\tau$ is a learnable temperature parameter (typically initialized around 0.07).

**Image-to-Text Loss (for image $i$):**
$$\mathcal{L}_i^{\text{img→txt}} = -\log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

This is the negative log probability that the correct text $t_i$ has the highest similarity to image $x_i$ among all $N$ texts in the batch.

**Text-to-Image Loss (symmetric):**
$$\mathcal{L}_i^{\text{txt→img}} = -\log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})}$$

**Total CLIP Loss:**
$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^{N} \left( \mathcal{L}_i^{\text{img→txt}} + \mathcal{L}_i^{\text{txt→img}} \right)$$

**Why Temperature $\tau$ Matters:**
- Small $\tau$ (e.g., 0.01): Sharper probability distribution, focuses on hard negatives, but can lead to training instability
- Large $\tau$ (e.g., 1.0): Softer distribution, easier optimization, but weaker discrimination
- CLIP learns $\tau$ during training, typically converging around 0.01-0.07

### Understanding the Contrastive Matrix

For a batch of 4 image-text pairs, the similarity matrix looks like:

```
                   Text_1  Text_2  Text_3  Text_4
              ┌─────────────────────────────────┐
    Image_1   │   ✓      ✗       ✗       ✗      │
    Image_2   │   ✗      ✓       ✗       ✗      │
    Image_3   │   ✗      ✗       ✓       ✗      │
    Image_4   │   ✗      ✗       ✗       ✓      │
              └─────────────────────────────────┘
    ✓ = Positive pair (maximize similarity)
    ✗ = Negative pair (minimize similarity)
```

With batch size $N$, each image has 1 positive and $N-1$ negatives. Larger batches provide harder negatives, improving representation quality. CLIP used batch sizes of 32,768.

### Projection and Normalization

Before computing similarity, embeddings are:

1. **Projected** to a common dimension:
$$\mathbf{z}^{\text{img}} = W_{\text{img}} \cdot \mathbf{h}^{\text{img}} + \mathbf{b}_{\text{img}}$$

2. **L2-Normalized** to lie on the unit hypersphere:
$$\hat{\mathbf{z}} = \frac{\mathbf{z}}{||\mathbf{z}||_2}$$

Normalization ensures cosine similarity equals dot product, simplifying computation and stabilizing training.

### Vision Encoder: Vision Transformer (ViT)

Most modern multimodal models use [[Vision Transformers]] (ViT) as the image encoder.

**Patch Embedding:**

An image $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ is divided into $N = \frac{HW}{P^2}$ patches of size $P \times P$.

Each patch is flattened and linearly projected:
$$\mathbf{z}_p^0 = \mathbf{x}_p \mathbf{E} + \mathbf{e}_{\text{pos}}$$

where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the projection matrix and $\mathbf{e}_{\text{pos}}$ is the positional embedding.

A learnable [CLS] token is prepended:
$$\mathbf{Z}^0 = [\mathbf{z}_{\text{CLS}}; \mathbf{z}_1^0; \mathbf{z}_2^0; ...; \mathbf{z}_N^0]$$

**Transformer Processing:**
$$\mathbf{Z}^l = \text{TransformerBlock}(\mathbf{Z}^{l-1})$$

The final [CLS] token $\mathbf{z}_{\text{CLS}}^L$ serves as the global image representation.

---

## Key Models and Architectures

### CLIP (Contrastive Language-Image Pre-training)

**OpenAI, 2021** - The foundational model that popularized joint embedding spaces.

| Aspect | Details |
|--------|---------|
| Training Data | 400M image-text pairs from internet (WIT dataset) |
| Image Encoders | ResNet-50/101, ViT-B/32, ViT-B/16, ViT-L/14 |
| Text Encoder | 12-layer, 512-dim Transformer (GPT-2 style) |
| Embedding Dim | 512 or 768 (depends on variant) |
| Batch Size | 32,768 |
| Zero-shot ImageNet | 76.2% (ViT-L/14@336px) |

**Key Innovations:**
- Natural language supervision (no fixed label set)
- Massive scale contrastive learning
- Prompt engineering for zero-shot classification

### ALIGN (A Large-scale ImaGe and Noisy text embedding)

**Google, 2021** - Similar to CLIP but with noisier, larger data.

| Aspect | Details |
|--------|---------|
| Training Data | 1.8B image-alt-text pairs (noisy) |
| Image Encoder | EfficientNet-L2 |
| Text Encoder | BERT-Large |
| Key Insight | Noise can be overcome with scale |

### SigLIP (Sigmoid Loss for Language-Image Pre-training)

**Google, 2023** - Improved contrastive objective.

Replaces softmax-based contrastive loss with **sigmoid loss**:
$$\mathcal{L} = -\frac{1}{N^2} \sum_{i,j} \log \sigma(y_{ij} \cdot S_{ij})$$

where $y_{ij} = 1$ for positive pairs, $y_{ij} = -1$ for negatives.

**Advantages:**
- No need for large batch sizes (can use smaller batches effectively)
- Better calibrated similarity scores
- Simpler distributed training

### BLIP-2 (Bootstrapping Language-Image Pre-training)

**Salesforce, 2023** - Efficient vision-language bridge.

```
┌───────────────┐
│ Frozen Image  │
│   Encoder     │
│   (ViT-G)     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Q-Former    │◄── Learnable Query Tokens
│  (Lightweight │
│   Querying    │
│  Transformer) │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Frozen LLM   │
│  (OPT/FlanT5) │
└───────────────┘
```

**Q-Former**: A small transformer that learns to extract relevant visual information for the LLM using 32 learnable query tokens. Only Q-Former is trained, keeping both vision and language models frozen.

### LLaVA (Large Language and Vision Assistant)

**Microsoft/Wisconsin, 2023** - Simple but effective visual instruction tuning.

Architecture:
1. **Vision Encoder**: CLIP ViT-L/14 (frozen or fine-tuned)
2. **Projection**: Simple linear or MLP layer
3. **LLM**: Vicuna/LLaMA (fine-tuned)

**Training Recipe:**
1. **Stage 1 (Feature Alignment)**: Train only the projection layer on image-caption pairs
2. **Stage 2 (Visual Instruction Tuning)**: Fine-tune the full model on instruction-following data

The simplicity of LLaVA (just a linear projection!) showed that a well-trained vision encoder + capable LLM + good instruction data is often sufficient.

### Flamingo

**DeepMind, 2022** - Few-shot multimodal learning.

Key innovation: **Perceiver Resampler** - compresses arbitrary-length visual features into a fixed number of visual tokens, enabling handling of multiple images/videos in context.

```
Variable-length         Fixed-length
Visual Features   →    Visual Tokens
 (N patches)           (64 tokens)
      │                     │
      └──► Perceiver ◄──────┘
           Resampler
           (Cross-attention with
            learnable queries)
```

---

## Training Strategies and Considerations

### Data Quality vs. Quantity

| Approach | Data Size | Data Quality | Examples |
|----------|-----------|--------------|----------|
| Curated | ~15M | High (human-verified) | COCO, Visual Genome |
| Web-scale noisy | 400M-5B | Low-Medium | CLIP WIT, LAION |
| Synthetic | Unlimited | Variable | Generated captions |

**Observation**: Web-scale noisy data + contrastive learning tends to outperform smaller curated datasets due to the diversity of concepts encountered.

### Batch Size Impact

Contrastive learning benefits enormously from large batch sizes:

| Batch Size | Negatives per Sample | Training Cost | Quality |
|------------|---------------------|---------------|---------|
| 256 | 255 | Low | Moderate |
| 4,096 | 4,095 | Medium | Good |
| 32,768 | 32,767 | Very High | Best |

**Workarounds for limited compute:**
- Gradient caching/accumulation
- Memory banks of past embeddings
- Distributed training across many GPUs
- SigLIP's sigmoid loss (batch-size independent)

### Resolution and Patch Size Trade-offs

For ViT-based encoders:

| Config | Patches (224px) | Patches (336px) | Compute | Detail |
|--------|-----------------|-----------------|---------|--------|
| ViT-B/32 | 49 | 121 | Low | Coarse |
| ViT-B/16 | 196 | 441 | Medium | Medium |
| ViT-L/14 | 256 | 576 | High | Fine |

Smaller patch size = more patches = finer detail but quadratically more compute in attention layers.

---

## Practical Applications

### Zero-Shot Image Classification

Convert classification into retrieval:

```python
# Pseudocode for zero-shot classification
class_prompts = ["a photo of a cat",
                 "a photo of a dog",
                 "a photo of a bird"]

text_embeddings = encode_text(class_prompts)  # [3, 512]
image_embedding = encode_image(test_image)     # [1, 512]

similarities = image_embedding @ text_embeddings.T  # [1, 3]
predicted_class = argmax(similarities)
```

**Prompt Engineering Matters:**
- "a photo of a {class}" works better than just "{class}"
- "a centered satellite photo of {class}" for aerial imagery
- Ensemble multiple prompts for robustness

### Cross-Modal Retrieval

```
Query: "sunset over mountains"
        │
        ▼
   Text Encoder
        │
        ▼
   [Query Vector]
        │
        ▼
   ┌────────────────────────────────────┐
   │  Image Database (pre-computed)     │
   │  [img_1] [img_2] ... [img_N]       │
   └────────────────────────────────────┘
        │
   Cosine Similarity Search (ANN)
        │
        ▼
   Top-K Results: 🌄 🏔️ 🌅
```

Approximate Nearest Neighbor (ANN) search with libraries like FAISS enables retrieval over billions of images in milliseconds.

### Visual Question Answering (VQA)

Modern VLMs handle VQA by conditioning text generation on both image and question:

```
Input: [Image Tokens] + "What color is the car?"
              │
              ▼
         VLM (e.g., LLaVA)
              │
              ▼
Output: "The car is red."
```

The joint embedding allows the model to ground language in visual evidence.

### Image Captioning

Autoregressive generation conditioned on image:
$$P(\text{caption} | \text{image}) = \prod_{t=1}^{T} P(w_t | w_{<t}, \mathbf{z}^{\text{img}})$$

---

## When to Use

| Use Case | Recommended Approach |
|----------|---------------------|
| Large-scale retrieval | Dual-encoder (CLIP, SigLIP) |
| Zero-shot classification | Dual-encoder with prompt tuning |
| Complex reasoning about images | Cross-attention VLM (GPT-4V, LLaVA) |
| Few-shot in-context learning | Flamingo-style architecture |
| Real-time applications | Smaller dual-encoder (CLIP ViT-B/32) |

### When NOT to Use

- **Pixel-precise tasks**: Segmentation, detection need additional heads (see [[SAM]], [[DETR]])
- **Fine-grained classification**: May need domain-specific fine-tuning
- **Tasks requiring 3D understanding**: Current models struggle with spatial reasoning
- **Counting objects**: Notorious failure mode for CLIP-style models

---

## Common Pitfalls

1. **Modality Gap**: Even after training, image and text embeddings occupy different sub-regions of the space. Direct interpolation may land in "dead zones."

2. **Compositionality Failures**: CLIP struggles with compositional concepts like "a red cube on a blue sphere" vs. "a blue cube on a red sphere." The bag-of-concepts tendency ignores relationships.

3. **Typographic Attacks**: CLIP can be fooled by text rendered in images. An image of an apple with "iPod" written on it gets classified as an iPod.

4. **Bias Amplification**: Web-scraped data contains societal biases that get encoded into the embedding space.

5. **Distribution Shift**: Performance degrades on domains far from web images (medical, satellite, microscopy). Domain-specific fine-tuning often necessary.

---

## Comparisons

| Model | Architecture | Training Objective | Strengths | Limitations |
|-------|--------------|-------------------|-----------|-------------|
| CLIP | Dual-encoder | Contrastive (InfoNCE) | Zero-shot, retrieval | No generation, compositionality |
| ALIGN | Dual-encoder | Contrastive | Scale tolerance | Similar to CLIP |
| BLIP-2 | Q-Former bridge | Contrastive + Generative | Efficient, modular | Fixed query count |
| LLaVA | Direct projection | Instruction tuning | Simple, effective | Needs instruction data |
| Flamingo | Perceiver + Gated XAttn | Next-token prediction | Few-shot, video | Complex architecture |
| GPT-4V | Proprietary | Unknown | Strongest reasoning | Closed source, cost |

---

## Current Research Directions

### Scaling Laws for Multimodal Models

Preliminary evidence suggests:
- Vision encoder quality matters more than size after a threshold
- LLM capability is the primary bottleneck for complex reasoning
- Data diversity trumps data size for generalization

### Video Understanding

Extending to temporal dimension:
- **Frame sampling**: Which frames to include?
- **Temporal attention**: How to model time?
- **Efficiency**: Video = many frames = expensive

Models like **VideoLLaVA**, **Video-ChatGPT** are early explorations.

### Unified Multimodal Models

Moving toward single models handling all modalities:
- Text, images, audio, video in one embedding space
- **ImageBind** (Meta): 6 modalities aligned through image pivots
- **Gemini**: Native multimodal from the ground up

### Reducing Hallucinations

VLMs often "hallucinate" objects not present in images. Active research on:
- Better training objectives
- RLHF for visual grounding
- Retrieval augmentation

---

## Resources

### Papers

- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN)](https://arxiv.org/abs/2102.05918) - Jia et al., 2021
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) - Li et al., 2023
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) - Liu et al., 2023
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) - Alayrac et al., 2022
- [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) - Zhai et al., 2023
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020
- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665) - Girdhar et al., 2023

### Articles & Tutorials

- [OpenAI CLIP Blog Post](https://openai.com/research/clip)
- [Lilian Weng's Survey on Multimodal Learning](https://lilianweng.github.io/posts/2022-06-09-vlm/)
- [Hugging Face CLIP Tutorial](https://huggingface.co/docs/transformers/model_doc/clip)

### Code Repositories

- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open source CLIP training
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Visual instruction tuning
- [LAION](https://laion.ai/) - Open datasets for multimodal training

### Videos

- [Yannic Kilcher - CLIP Paper Explained](https://www.youtube.com/watch?v=T9XSU0pKX2E)
- [AI Coffee Break - Vision Transformers](https://www.youtube.com/watch?v=TrdevFK_am4)

---

**Back to**: [[LLMs & Generative AI]]
