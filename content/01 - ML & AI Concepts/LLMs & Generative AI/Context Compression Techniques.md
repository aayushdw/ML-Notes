# Context Compression Techniques

## Overview

Context compression refers to methods that reduce the length of input prompts while preserving the essential information needed for accurate LLM responses. Context compression operates at the **input level**, aiming to fit more meaningful content within a model's [[Context Window Fundamentals|context window]] or reduce computational costs by processing fewer tokens.

The core motivation: API costs scale with token count, latency increases with sequence length, and the "Lost in the Middle" problem means very long contexts often hurt performance anyway. 


## Why Compress Context?

Consider a [[Naive RAG Pipeline|RAG pipeline]] that retrieves 10 documents of 500 tokens each. That is 5,000 tokens of context before we even add the user query or system prompt. Most of those tokens are filler words, redundant phrases, or marginally relevant content. Compression identifies and removes the noise.

## The Compression Spectrum

![[Context Compression Techniques 2025-12-30 14.58.36.excalidraw.svg]]

### The Information Bottleneck Intuition

Think of compression as finding the **minimal sufficient representation**. If you have a 2,000-token document and someone asks "What year was the company founded?", perhaps only 15 tokens in that document actually matter. Compression techniques try to identify and retain those 15 tokens (or their semantic equivalent).

## Main Techniques

### 1. LLMLingua Family

The LLMLingua approach uses a small "compressor" LLM to identify which tokens can be dropped without losing meaning.

#### LLMLingua (Original)

- Uses perplexity from a small model (e.g., GPT-2, LLaMA-7B) as an importance signal
- **High perplexity tokens** = surprising/informative = keep them
- **Low perplexity tokens** = predictable/redundant = safe to remove
- Achieves **up to 20x compression** with minimal performance loss on certain benchmarks

**Process**:
1. Run the context through a small LLM
2. Compute token-level perplexity scores
3. Remove tokens below a threshold (or keep top-k% by perplexity)
4. Feed the compressed prompt to the target LLM

#### LongLLMLingua

Extension for [[Context Window Fundamentals|long-context]] scenarios. Adds:
- **Question-aware compression**: Considers the query when deciding what to keep
- **Document reordering**: Moves important content to the beginning and end (addressing "Lost in the Middle")
- **Contrastive perplexity**: Measures how surprising a token is *given the question* vs. in isolation

#### LLMLingua-2

Uses a trained classifier (small BERT-like model) instead of perplexity:
- Binary classification: keep or discard each token
- Trained on distillation data from GPT-4 judgments
- Faster than perplexity-based methods (no autoregressive forward pass needed)

### 2. Selective Context

A simpler approach that uses **self-information** (negative log probability) to filter tokens:

$$
I(x_i) = -\log P(x_i | x_{<i})
$$

Tokens with low self-information (highly predictable given prior context) are removed. The intuition: if you can predict a word perfectly from context, including it adds no new information.

**Algorithm**:
1. Compute conditional probabilities for each token using a causal LM
2. Calculate self-information scores
3. Apply a threshold or keep a fixed percentage
4. Concatenate remaining tokens

### 3. Gist Tokens / Gisting

Gist tokens are **learned virtual tokens** that summarize longer contexts into a fixed number of embeddings.

```
┌──────────────────────────────────────────────────────────────────┐
│  Original: "The quick brown fox jumps over the lazy dog"         │
│                              ↓                                   │
│  Gist Compression (k=2 gist tokens)                              │
│                              ↓                                   │
│  [GIST_1] [GIST_2]  ←  Dense vectors encoding the sentence       │
│                                                                  │
│  These 2 "virtual tokens" replace 9 real tokens                  │
└──────────────────────────────────────────────────────────────────┘
```

**Training**:
- Fine-tune model to produce useful gist embeddings
- Gist tokens are prepended to the actual input
- Model learns to condition on gist tokens for downstream tasks

**Relation to [[P-Tuning|Prefix-Tuning]]**: Gisting is conceptually similar, but the goal is compression rather than task adaptation. The "prefix" here summarizes the context.

### 4. AutoCompressors

AutoCompressors take gisting further by training the model to compress its own context iteratively:

1. Process the first segment, generate "summary vectors"
2. Prepend summary vectors to the next segment
3. Repeat, accumulating compressed representations
4. Final summary vectors encode the entire document

This allows processing documents longer than the context window by compressing as you go.

**Key Equation** (conceptual):

$$
\text{summary}_{t} = f_\theta(\text{segment}_t, \text{summary}_{t-1})
$$

where $f_\theta$ is the autocompressor model that takes a text segment and the previous summary, producing a new summary.

### 5. Summarization-Based Compression

The most straightforward approach: use an LLM to summarize retrieved documents before including them in the prompt.

**Two Variants**:
- **Extractive**: Select important sentences verbatim
- **Abstractive**: Generate a condensed paraphrase

**Trade-offs**:

| Aspect | Extractive | Abstractive |
|:-------|:-----------|:------------|
| Faithfulness | High (original text) | Risk of hallucination |
| Compression ratio | Moderate (sentence-level) | High (can be very brief) |
| Latency | Low | Higher (LLM call) |
| Cost | Low | Additional inference cost |

### 6. RECOMP (Retrieval Compression)

Designed specifically for [[Naive RAG Pipeline|RAG pipelines]]. Two components:

1. **Extractive Compressor**: Selects relevant sentences from retrieved documents
2. **Abstractive Compressor**: Generates a summary conditioned on the query

The compressor is trained to produce outputs that maximize downstream QA accuracy, not just general summarization quality.

### 7. Nugget-Based Compression

Identifies atomic "nuggets" of information in a document:
- Each nugget is a self-contained fact
- Nuggets are scored for relevance to the query
- Only high-scoring nuggets are included

This is more semantic than token-level pruning, operating at the fact/claim level.

## Mathematical Foundation

### Perplexity-Based Token Selection

For a token sequence $x_1, x_2, ..., x_n$, the perplexity of token $x_i$ given its prefix is:

$$
\text{PPL}(x_i) = \exp\left(-\log P(x_i | x_{<i})\right) = \frac{1}{P(x_i | x_{<i})}
$$

**Selection Rule**: Keep token $x_i$ if $\text{PPL}(x_i) > \tau$ (threshold).

Tokens with high perplexity are "surprising" given context, meaning they carry more information.

### Compression Ratio

$$
\text{Compression Ratio} = \frac{|\text{Original Tokens}|}{|\text{Compressed Tokens}|}
$$

A ratio of 10x means you reduced a 1,000-token input to 100 tokens.

### Information-Theoretic View

Compression fundamentally trades off **rate** (number of bits/tokens) against **distortion** (information loss). The optimal compressed representation minimizes:

$$
\mathcal{L} = R + \lambda D
$$

where $R$ is the length of the compressed representation and $D$ is a distortion measure (e.g., performance drop on downstream tasks). $\lambda$ controls the rate-distortion trade-off.

## Practical Application

### When to Use

- **High API costs**: Compression can cut token usage by 50-90%
- **Latency-sensitive applications**: Fewer tokens = faster inference
- **Long document QA**: Fitting multiple documents in context
- **RAG pipelines with many retrieved chunks**: Compress before generation

### When NOT to Use

- **Short prompts**: Overhead of compression outweighs benefits
- **Tasks requiring exact wording**: Legal documents, code generation from specs
- **Low-resource languages**: Compression models may perform poorly
- **When you need full auditability**: Hard to debug what was removed

### Common Pitfalls

1. **Over-compression**: Removing too much loses critical information
2. **Domain mismatch**: Compressor trained on news may fail on medical text
3. **Ignoring query context**: Generic compression loses query-relevant details
4. **Cascading errors**: If compressor misses something, LLM cannot recover

### Trade-offs and Calculations

**Latency Analysis**:
- Compression adds overhead (small model forward pass)
- But reduces main LLM processing time
- Break-even depends on: compression ratio, relative model sizes, context length

**Example Calculation**:
- Original: 4,000 tokens at \$0.01/1K = \$0.04 per request
- Compressed (5x): 800 tokens = \$0.008 per request
- Savings: 80% cost reduction

**Quality Degradation**:
Typical benchmarks show:
- 2-4x compression: <1% accuracy drop
- 5-10x compression: 2-5% accuracy drop
- 10-20x compression: 5-15% accuracy drop (varies heavily by task)

## Comparisons

| Method | Compression Type | Typical Ratio | Requires Training | Query-Aware |
|:-------|:-----------------|:--------------|:------------------|:------------|
| LLMLingua | Hard (token pruning) | 5-20x | No (uses pretrained LM) | Optional |
| LLMLingua-2 | Hard (token pruning) | 5-15x | Yes (classifier) | Yes |
| Selective Context | Hard (token pruning) | 2-5x | No | No |
| Gist Tokens | Soft (embeddings) | Variable | Yes | Depends |
| RECOMP | Hard (sentence selection) | 3-10x | Yes | Yes |
| Summarization | Hard (abstractive) | 5-20x | Optional | Optional |

### Comparison with Related Concepts

| Concept | Goal | Operates On |
|:--------|:-----|:------------|
| Context Compression | Reduce input length | Prompt/retrieved docs |
| [[Token Optimization]] | Faster inference | Attention/KV cache |
| [[Chunking Strategies]] | Better retrieval | Document indexing |
| [[Re-ranking]] | Improve relevance | Retrieved results |

## Resources

### Papers
- [LLMLingua: Compressing Prompts for Accelerated Inference](https://arxiv.org/abs/2310.05736) - Original LLMLingua paper
- [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios](https://arxiv.org/abs/2310.06839) - Long-context extension
- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) - Classifier-based approach
- [Compressing Context to Enhance Inference Efficiency of Large Language Models](https://arxiv.org/abs/2310.06201) - Selective Context paper
- [Gisting: Training Language Models to Compress Prompts](https://arxiv.org/abs/2304.08467) - Gist tokens
- [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408) - RAG-specific compression

### Articles
- [LLMLingua GitHub Repository](https://github.com/microsoft/LLMLingua) - Microsoft's official implementation
- [Prompt Compression and Query Optimization](https://blog.langchain.dev/prompt-compression/) - LangChain blog post

### Videos
- [LLMLingua Explained (Yannic Kilcher)](https://www.youtube.com/watch?v=2Qi1rXqVoWs) - Walkthrough of the method

---
**Back to**: [[02 - LLMs & Generative AI Index]]
