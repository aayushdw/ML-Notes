## Overview

Embeddings are dense, fixed-dimensional vector representations of data (text, images, audio) that capture semantic meaning. In the context of [[01 - RAG Index|RAG pipelines]], embeddings enable semantic search by converting both documents and queries into a shared vector space where similarity can be measured geometrically.

The core insight is that semantically similar content should have similar embeddings:
- "The quick brown fox" ≈ "A fast auburn fox" (close in vector space)
- "The quick brown fox" ≠ "Quarterly earnings report" (far apart in vector space)

$$\text{Text} \xrightarrow{f_\theta} \mathbf{v} \in \mathbb{R}^d$$

Where $f_\theta$ is the embedding model (a neural network with learned parameters $\theta$), and $d$ is the embedding dimension.


## How Embeddings Work

### From Words to Vectors

Embedding models learn to map text to vectors during training. The training process optimizes the model so that semantically similar texts produce similar vectors.

**Training Objective (Contrastive Learning)**:
Given a query $q$ and a positive document $d^+$ (relevant), and negative documents $d^-$ (irrelevant), the model learns to use below loss function to:
- Maximize similarity between $q$ and $d^+$
- Minimize similarity between $q$ and $d^-$

$$\mathcal{L} = -\log \frac{e^{sim(q, d^+)/\tau}}{\sum_{d \in \{d^+, d^-\}} e^{sim(q, d)/\tau}}$$

Where $\tau$ is a temperature parameter and $sim(\cdot, \cdot)$ is typically cosine similarity.

### Understanding the Vector Space

Embeddings encode semantic relationships in geometric relationships:

**Linear Relationships**:
The famous Word2Vec analogy: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$

This property extends to sentence embeddings, though less perfectly. Related concepts cluster together, and vector arithmetic can sometimes discover relationships.

**Clustering by Topic**:
Documents about similar topics naturally cluster:
- Legal documents form one cluster
- Medical research forms another cluster
- Sports news forms yet another

This clustering is what enables retrieval: a query about "contract law" will be close to legal documents in vector space.


## Embedding Model Architectures

### Bi-Encoder Architecture

The standard architecture for retrieval embeddings. Encodes queries and documents independently.

![[Embeddings 2026-01-12 00.00.00.excalidraw.svg]]

**Characteristics**:
- Query and document encoded separately
- Document embeddings can be pre-computed and cached
- Fast at query time (just encode query, then vector search in the [[Vector Databases]])
- No cross-attention between query and document

**Training**: Typically uses contrastive learning with in-batch negatives.

### Asymmetric vs Symmetric Models

| Type | Query Style | Document Style | Use Case |
|:-----|:------------|:---------------|:---------|
| **Symmetric** | Same as document | Same as query | Semantic similarity, deduplication |
| **Asymmetric** | Short questions | Long passages | Q&A retrieval, RAG |

**Asymmetric models** are trained specifically for the query-document retrieval task. The query encoder learns to represent questions, while the document encoder learns to represent answer-containing passages.

Most production RAG systems use asymmetric models (OpenAI, Cohere, BGE).


### Choosing an Embedding Model

**Factors to Consider**:
1. **Quality**: How well does it perform on your specific domain/task?
2. **Dimensionality**: Higher dimensions = more expressive, but more storage/compute
3. **Max Tokens**: Can it handle your chunk sizes?
4. **Latency**: How fast is inference?
5. **Cost**: API pricing or self-hosting compute
6. **Multilingual**: Does it support your languages?


## MTEB Benchmark

The **Massive Text Embedding Benchmark (MTEB)** is the standard benchmark for evaluating embedding models across diverse tasks.

### Task Categories

| Task | Description | Relevance to RAG |
|:-----|:------------|:-----------------|
| **Retrieval** | Find relevant documents for a query | Direct RAG relevance |
| **Semantic Textual Similarity (STS)** | Score similarity between sentence pairs | Related to retrieval quality |
| **Classification** | Categorize text into classes | Less relevant |
| **Clustering** | Group similar documents | Document organization |
| **Reranking** | Reorder candidates by relevance | Post-retrieval refinement |
| **Pair Classification** | Binary similarity decisions | Deduplication |


**For RAG, prioritize Retrieval scores**. The benchmark reports:
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **Recall@k**: Percentage of relevant documents in top-k

Leaderboard: https://huggingface.co/spaces/mteb/leaderboard

## Dimensionality and Its Effects

### What Dimensionality Means
The embedding dimension $d$ determines the size of the vector:
- `text-embedding-3-small`: $d = 1536$
- `all-MiniLM-L6-v2`: $d = 384$

Higher dimensions can theoretically capture more nuanced semantic information, but with diminishing returns.

### Dimension Reduction (Matryoshka Embeddings)
Some modern models (OpenAI `text-embedding-3-*`, Nomic) support [Matryoshka Representation Learning (MRL)](https://medium.com/@zilliz_learn/matryoshka-representation-learning-explained-the-method-behind-openais-efficient-text-embeddings-a600dfe85ff8) , which allows truncating embeddings to smaller dimensions while retaining most quality.

The model is trained so that the first $k$ dimensions are a valid embedding on their own. Truncate from 1536 → 512 dimensions and still get useful embeddings.

**Benefits**:
- Reduce storage by 3x (1536 → 512)
- Faster similarity computation
- Minimal quality loss (typically 1-3% on benchmarks)


## Practical Application

### Batch Embedding Calls

For large-scale indexing, batch multiple texts per API call:
```python
# Inefficient: One call per document
for doc in documents:
    embedding = embed(doc)  # N API calls

# Efficient: Batch calls
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    embeddings = embed(batch)  # N/100 API calls
```



## Domain Adaptation

### The Domain Shift Problem
General embedding models are trained on web text (Wikipedia, news, forums). When applied to specialized domains, performance degrades:

| Domain     | Challenge                                                  |
| :--------- | :--------------------------------------------------------- |
| Legal      | Specific terminology ("tort", "habeas corpus", "estoppel") |
| Medical    | Technical vocabulary ("acute myocardial infarction")       |
| Scientific | Jargon, abbreviations, formulas                            |
| Code       | Syntax, variable names, non-natural language               |

### Strategies for Domain Adaptation

#### 1. Fine-Tuning (Best Quality, Most Effort)
Train on domain-specific data:
- Collect query-document pairs from your domain
- Fine-tune using contrastive loss
- Requires labeled data (or synthetic generation)

Frameworks: Sentence Transformers, [[LoRA (Low-Rank Adaptation)]]

#### 2. Use Domain-Specific Models
Pre-trained models for specific domains:
- **Legal**: Legal-BERT embeddings
- **Medical**: PubMedBERT, BiomedCLIP
- **Scientific**: SciBERT, SPECTER2
- **Code**: CodeBERT, StarCoder embeddings

#### 3. Hybrid Retrieval
Combine embeddings with [[BM25]] keyword search:
- Embeddings catch semantic matches
- BM25 catches exact terminology
- [[Hybrid Search]] combines both for robustness
- This is often the simplest mitigation for domain shift.


## Multilingual Embeddings

### Cross-Lingual Retrieval
Multilingual models map text from different languages into the same vector space:
- Query in English → retrieve documents in French, German, Chinese
- Single index for all languages

### Leading Multilingual Models

| Model | Languages | Notes |
|:------|:----------|:------|
| **BGE-M3** | 100+ | Excellent multilingual, open-source |
| **Cohere embed-v3** | 100+ | API-based, strong performance |
| **E5-multilingual** | 100+ | Open-source |
| **OpenAI text-embedding-3-**** | 100+ | Good multilingual support |

### Considerations
- Cross-lingual performance is typically 5-15% lower than monolingual
- Some language pairs work better than others (related languages transfer better)
- For critical applications, consider language-specific models

## Token Limits and Long Documents

### The Context Window Problem
Most embedding models have limited context windows:
- `all-MiniLM-L6-v2`: 256 tokens
- `BGE-base`: 512 tokens
- `OpenAI text-embedding-3-small`: 8191 tokens

Text beyond the limit is truncated, losing information.

### Strategies for Long Documents

#### 1. Chunk and Embed (Standard Approach)
Split documents into chunks that fit the context window. See [[Chunking Strategies]].

**Trade-off**: Loses document-level context.

#### 2. Use Long-Context Models
Choose models with larger windows:
- `nomic-embed-text`: 8192 tokens
- `GTE-large`: 8192 tokens
- `jina-embeddings-v2`: 8192 tokens

**Trade-off**: Longer context = slower inference, higher cost.

#### 3. Hierarchical Embeddings
Embed at multiple granularities:
- Sentence-level embeddings for precision
- Paragraph-level for broader context
- Document-level for theme matching

Query against all levels, merge results.

#### 4. Late Chunking
Newer technique where:
1. Run the full document through the transformer (up to max tokens)
2. Pool token embeddings into chunk embeddings afterward

**Benefit**: Chunks retain context from surrounding text.

**Implementations**: LlamaIndex, some embedding providers


## Common Pitfalls

### 1. Embedding Model Mismatch
**Problem**: Using different models for indexing vs. querying.

**Why It Breaks**: Vector spaces are incompatible. Similarity scores become meaningless.

**Solution**: Track which model created each embedding. Store model name as metadata.

### 2. Ignoring Instruction Prefixes
**Problem**: Using instruction-tuned models without the required prefixes.

**Why It Matters**: Model was trained with specific formats. Without them, embeddings are suboptimal.

**Solution**: Check model documentation. BGE, E5, and others require specific prefixes.

### 3. Truncation Without Awareness
**Problem**: Long text silently truncated.

**Symptoms**: Documents that should be similar are not. Key information at the end of chunks is lost.

**Solution**: Monitor token counts. Design chunking to stay within limits. Consider overlap.

### 4. Not Normalizing Embeddings
**Problem**: Using cosine similarity with non-normalized embeddings.

**Why It Matters**: Cosine similarity assumes unit vectors. Results may be incorrect.

**Solution**: Check model documentation. Normalize if not done by default.

### 5. Overfitting to Benchmarks
**Problem**: Choosing model purely based on MTEB scores.

**Reality**: Benchmark performance does not always transfer to your specific domain.

**Solution**: Evaluate on your own data. Create a small test set of queries and relevant documents.


## Comparisons

### Embedding vs TF-IDF/BM25

| Aspect | Embeddings (Dense) | TF-IDF/BM25 (Sparse) |
|:-------|:-------------------|:---------------------|
| **Representation** | Dense vectors ($d \approx 384-3072$) | Sparse vectors ($d = $ vocabulary size) |
| **Semantic Understanding** | Yes (synonyms, paraphrases) | No (exact matches only) |
| **Training** | Required (pre-trained models) | None (statistical formula) |
| **Storage** | Fixed size per document | Variable (depends on document length) |
| **Exact Term Matching** | Weak | Strong |
| **Novel Vocabulary** | Handles via subword tokenization | Fails on unseen terms |
| **Best For** | Semantic similarity | Keyword matching, domain terms |

### Embedding Model Comparison

| Model | Pros | Cons |
|:------|:-----|:-----|
| **OpenAI** | High quality, easy API | Cost, vendor lock-in |
| **Cohere** | Multilingual, compression | Cost |
| **BGE-M3** | Open-source, hybrid, multilingual | Self-hosting complexity |
| **E5-large-v2** | Strong open-source | Shorter context (512) |
| **all-MiniLM-L6-v2** | Fast, tiny | Lower quality, short context |


## Advanced Topics

### Sparse-Dense Hybrid Embeddings
Models like **BGE-M3** and **SPLADE** output both dense and sparse representations:

**Dense**: Traditional semantic embedding (1024 dims)
**Sparse**: Learned term weights (vocabulary-size sparse vector)

**Benefits**:
- Combines semantic understanding with keyword matching
- Single model for [[Hybrid Search]]
- No separate BM25 index needed


### Embedding Quantization
Reduce storage and speed up search by quantizing embeddings:

| Type | Original | Quantized | Memory Savings |
|:-----|:---------|:----------|:---------------|
| **float32** | 4 bytes/dim | Baseline | 0% |
| **float16** | 4 bytes/dim | 2 bytes/dim | 50% |
| **int8** | 4 bytes/dim | 1 byte/dim | 75% |
| **binary** | 4 bytes/dim | 1 bit/dim | 97% |

Most [[Vector Databases]] support quantization options.


## Resources

### Documentation
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Embed Documentation](https://docs.cohere.com/docs/embeddings)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Text Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)
- [LlamaIndex: Embeddings](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/)

### Papers
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) - DPR
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) - E5
- [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity](https://arxiv.org/abs/2402.03216)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)

### Benchmarks
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Standard embedding benchmark
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Zero-shot retrieval benchmark

---

**Back to**: [[01 - RAG Index]]
