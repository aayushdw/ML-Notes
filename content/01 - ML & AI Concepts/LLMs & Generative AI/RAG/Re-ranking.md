## Overview

**Re-ranking** is a post-retrieval refinement step that re-scores and reorders the top-K candidates retrieved by the initial retrieval system. Instead of relying solely on the first retrieval pass (which uses approximate/fast similarity metrics), re-ranking applies a more sophisticated, expensive model to re-evaluate and reorder results for higher precision.

**Core Insight**: The initial retriever is optimized for *recall* (finding all potentially relevant documents quickly), often using fast approximate methods. Re-ranking is optimized for *precision* (finding the most relevant documents from the candidates). This two-stage approach balances speed and accuracy.

### Why Re-ranking Matters

From the [[01 - RAG Index#Advanced Retrieval|Advanced Retrieval]] phase:
- **Imperfect Initial Retrieval**: Dense retrievers (vector search) and sparse retrievers (BM25) often rank relevant documents below irrelevant ones
- **Lost in the Middle**: More relevant documents may be ranked 3rd or 4th, while the LLM uses only the top results; re-ranking ensures truly relevant docs appear first
- **Noise Reduction**: Retrieved results often include false positives; re-ranking filters these out
- **Quality Improvement**: Using a more sophisticated model for ranking can significantly improve downstream answer quality

### The Two-Stage Retrieval Paradigm

![[Re-ranking 2026-01-10 13.14.33.excalidraw.svg]]


## Core Concept: Cross-Encoder vs Bi-Encoder

### Bi-Encoder - Initial Retrieval

**Architecture**: Separately encode the query and documents, then compute similarity (usually dot product or cosine).

```
Query: "How do I fix a bug?"
    ↓ (Encoder 1)
    [vector: 768 dims]
                          Similarity Score
                                  ↑
Document: "Debugging techniques in Python"    [vector: 768 dims] ← (Encoder 2)
```

**Characteristics**:
- Compute similarities independently
- Score = similarity(Q_embedding, D_embedding)
- Very fast for retrieval (can pre-compute all document embeddings)
- Works at scale (billions of documents)
- **Problem**: Ignores inter-token interactions between query and document

**Example Models**: Sentence-BERT, DPR (Dense Passage Retrieval)


### Cross-Encoder - Re-ranking

**Architecture**: Encode query and document *together* as input, produce a relevance score.

```
[CLS] How do I fix a bug? [SEP] Debugging techniques in Python [SEP]
    ↓ (Single Encoder)
    [Multi-head Self-Attention across all tokens]
    ↓
    Relevance Score: 0.92
```

**Characteristics**:
- Encode query and document as a *single sequence*
- Score = relevance(query + document_pair)
- Can attend to interactions between query and document tokens
- Slower (must score each candidate individually; no pre-computation)
- **Advantage**: Much more accurate relevance assessment

**Example Models**: BERT-base-uncased fine-tuned for MS MARCO, `cross-encoder/mmarco-MiniLMv2-L12-H384-v1`


### Key Difference in Scoring

**Bi-Encoder**:
```
score(q, d) = cos(encode_q(q), encode_d(d))
```
- Independent encodings → fast but limited interaction
- Query and document don't influence each other's representation

**Cross-Encoder**:
```
score(q, d) = cross_encode([q, d])
```
- Joint encoding → slower but rich interaction
- Tokens in document attend to query tokens and vice versa
- Can understand nuanced relevance relationships


## Re-ranking Approaches

### 1. Pointwise Re-ranking (Most Common)

**Concept**: Score each retrieved document independently to get a relevance probability.
**How It Works**:
```
Retrieved documents: [doc1, doc2, doc3, ..., docK]
    ↓
Cross-encoder scores each:
  score(q, doc1) = 0.85
  score(q, doc2) = 0.72
  score(q, doc3) = 0.91
  ...
    ↓
Re-rank by score (highest first):
  doc3 (0.91), doc1 (0.85), doc2 (0.72), ...
```

**Strengths**:
- Simple and interpretable
- Score = probability of relevance (0-1)
- Easy to filter by threshold ("only include docs with score > 0.7")
- Works well for most use cases

**Best For**:
- General Q&A systems
- When interpretable relevance scores needed
- Filtering and thresholding decisions


### 2. Pairwise Re-ranking

**Concept**: Score pairs of documents relative to each other, then derive a total order.
**How It Works**:
```
Compare documents pairwise:
  compare(doc1, doc2) → doc1 is more relevant
  compare(doc1, doc3) → doc3 is more relevant
  compare(doc2, doc3) → doc3 is more relevant
    ↓
Derive total order: doc3 > doc1 > doc2
```

**Strengths**:
- Can be more accurate than pointwise (uses comparison information)
- Better at handling ranking tasks where relative order matters most

**Limitations**:
- Requires pairwise training data (expensive to create)
- Doesn't provide absolute relevance scores
- O(n²) comparisons for n documents

**Best For**:
- Large-scale ranking systems (search engines)
- When you need perfect ordering, not just relevance scores
- Systems with access to pairwise training data


### 3. Hybrid Re-ranking

Combine multiple re-ranking signals (cross-encoder score + BM25 + metadata filters + LLM judgment).

**Example**:
```python
def hybrid_rerank(query, candidates, cross_encoder, bm25, metadata_filters):
    scores = {}

    for doc in candidates:
        # Score 1: Cross-encoder relevance (0-1)
        ce_score = cross_encoder.score([query, doc]) * 0.6

        # Score 2: BM25 keyword match (normalized to 0-1)
        bm25_score = bm25.score(query, doc) * 0.2

        # Score 3: Metadata match (is it recent, from trusted source, etc.)
        metadata_score = check_metadata(doc) * 0.2

        total_score = ce_score + bm25_score + metadata_score
        scores[doc] = total_score

    reranked = sorted(scores, key=scores.get, reverse=True)
    return reranked
```

**Strengths**:
- Combines different signal sources
- More robust to individual signal failures
- Can tune weights per use case
- Balance between different relevance aspects

**Limitations**:
- More complex to implement and maintain
- Weight tuning is tricky (requires data)
- Harder to debug when performance degrades

**Best For**:
- Production systems with diverse requirements
- Systems with different document types


## Integration with the RAG Pipeline

Re-ranking appears at a critical juncture, after initial retrieval but before LLM generation:

![[Re-ranking 2026-01-10 13.31.36.excalidraw.svg]]

### Why This Ordering Matters:
1. **Query Transformation** → Makes query better match document vocabulary
2. **Initial Retrieval** → Casts a wide net, prioritizes recall
3. **Re-ranking** → Filters and reorders for precision (THIS STAGE)
4. **LLM** → Generates using the best documents

Without re-ranking, the LLM works with whatever the retriever found, which may include noise or miss relevant documents ranked 11-100.

## Trade-offs & Decision Framework

### Latency vs Accuracy

**No Re-ranking** (Fastest):
- Latency: ~50ms (just retrieval)
- Accuracy: Lower (uses fast approximate ranking)
- Use when: Latency is critical (< 100ms)

**Fast Re-ranking** (e.g., MiniLM):
- Latency: ~200ms (retrieval + rerank ~50 docs)
- Accuracy: Medium-High
- Use when: Balance needed for interactive systems

**Slow Re-ranking** (Large BERT, LLM):
- Latency: ~1-5 seconds
- Accuracy: Highest
- Use when: Accuracy matters more than latency (offline, batch)


![[Re-ranking 2026-01-10 13.34.43.excalidraw.svg]]



## Sample Implementation Patterns

### Pattern 1: Basic Cross-Encoder Re-ranking

```python
def rag_retrieval_with_reranking(query, vector_store, cross_encoder, top_k=10):
    # Stage 1: Fast retrieval (recall-oriented)
    initial_results = vector_store.similarity_search(query, top_k=100)

    # Stage 2: Slow re-ranking (precision-oriented)
    pairs = [[query, doc.content] for doc in initial_results]
    scores = cross_encoder.predict(pairs)

    # Stage 3: Reorder
    ranked = sorted(
        zip(initial_results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-K after reranking
    return [doc for doc, score in ranked[:top_k]]
```


### Pattern 2: Selective Re-ranking (Cost Optimization)

```python
def selective_rerank(query, vector_store, cross_encoder, threshold=0.7):
    # Retrieve more than needed initially
    candidates = vector_store.similarity_search(query, top_k=50)

    # Rerank only the top candidates (not all 50)
    to_rerank = candidates[:20]  # Rerank only top 20

    pairs = [[query, doc.content] for doc in to_rerank]
    scores = cross_encoder.predict(pairs)

    # Keep originals that weren't reranked, but append reranked ones
    reranked = sorted(zip(to_rerank, scores), key=lambda x: x[1], reverse=True)

    # Filter by threshold
    filtered = [doc for doc, score in reranked if score >= threshold]

    return filtered
```


### Pattern 3: Multi-stage Re-ranking

```python
def multi_stage_rerank(query, vector_store, fast_model, slow_model):
    # Stage 1: Initial retrieval
    candidates = vector_store.similarity_search(query, top_k=100)

    # Stage 2a: Fast re-ranking (filter down to top 20)
    fast_scores = fast_model.predict([[query, doc.content] for doc in candidates])
    fast_ranked = sorted(
        zip(candidates, fast_scores),
        key=lambda x: x[1],
        reverse=True
    )
    top_20 = [doc for doc, score in fast_ranked[:20]]

    # Stage 2b: Slow re-ranking (precise ranking of top 20)
    slow_scores = slow_model.predict([[query, doc.content] for doc in top_20])
    slow_ranked = sorted(
        zip(top_20, slow_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in slow_ranked]
```

This pattern balances cost and accuracy.
- Only top 20 see the slow model


## Evaluation & Metrics

### Key Metrics for Re-ranking

#### NDCG@K (Normalized Discounted Cumulative Gain)
- Measure: How good is the ranking compared to ideal ranking?
- Formula: $NDCG@K = \frac{DCG@K}{IDCG@K}$
- Range: 0-1 (1 is perfect ranking)
- Gold standard for ranking quality

**DCG (Discounted Cumulative Gain)**: Sum of relevance scores, with lower-ranked documents receiving diminishing credit:

$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}$$

Where $rel_i$ = relevance score at position $i$. 

**IDCG (Ideal DCG)**: The *best possible* DCG, computed by sorting all documents by relevance (highest first) and calculating DCG on this ideal ordering. Provides the theoretical ceiling for normalization.

#### MRR@K (Mean Reciprocal Rank):
- Measure: On average, how early is the first relevant result?
- Formula: `MRR = 1/rank_of_first_relevant`
- Best for: Single-answer scenarios (FAQ, entity lookup)

#### MAP@K (Mean Average Precision)
- Measure: Precision at each relevant document
- Good for: Evaluating comprehensive answer retrieval

#### Recall@K:
- Measure: Out of all relevant documents, what % did we retrieve?
- Formula: `relevant_retrieved / total_relevant`


## Common Pitfalls ( and  Potential Solutions )

### Over-Relying on Re-ranking

**Problem**: Using re-ranking to compensate for poor initial retrieval.
- If retrieval gets only 50% recall, re-ranking can't fix this (can't rerank what wasn't retrieved)

**Solution**:
- First optimize initial retrieval ([[Hybrid Search]], [[Query Transformations]])
- Use re-ranking as refinement, not rescue

### Model-Data Mismatch

**Problem**: Using cross-encoder trained on MS MARCO for medical documents.
- Mismatch between training data and target domain

**Solution**:
- Fine-tune on domain-specific data if possible
- Use small cross-encoder (faster to fine-tune)
- Fall back to ensemble methods if fine-tuning unavailable

### Computational Overhead

**Problem**: Re-ranking 1000 candidates with large BERT model = 10+ seconds.
- Defeats the purpose of fast retrieval

**Solution**:
- Multi-stage: Fast model on 100, then slow model on 20
- Batch processing to utilize GPU
- Use lighter models (MiniLM, TinyBERT)
- Selective re-ranking: only rerank top-K from initial retrieval

### Score Mis-calibration

**Problem**: Cross-encoder outputs [0.92, 0.78, 0.45] but you don't know what these mean absolutely (just relative).
- Can't use raw scores for filtering/thresholding

**Solution**:
- Use models trained with proper calibration (temperature scaling)
- Don't rely on absolute score values; use relative ranking
- Test thresholds empirically on your data


## Comparison with Alternatives
Latency numbers are an estimation here (obviously) just the provide a vague idea.

| Approach                  | Latency*  | Accuracy  | Complexity | Cost   | Best For              |
| :------------------------ | :-------- | :-------- | :--------- | :----- | :-------------------- |
| **No Re-ranking**         | 50ms      | Low       | 0          | $0     | Real-time, low stakes |
| **Cross-Encoder (Fast)**  | 200ms     | Medium    | 1          | Low    | Balanced systems      |
| **Cross-Encoder (Large)** | 1-2s      | High      | 1          | Medium | Accuracy-critical     |
| **LLM Re-ranking**        | 2-5s      | Very High | 2          | High   | Complex relevance     |
| **Hybrid**                | 300-500ms | High      | 3          | Medium | Production systems    |
| **Learning-to-Rank**      | 100-300ms | Very High | 4          | High   | Search engines        |


## Integration with Query Transformations and Hybrid Search

![[Re-ranking 2026-01-10 13.44.02.excalidraw.svg]]


## See Also
- [[01 - RAG Index#Advanced Retrieval|Advanced Retrieval Phase]] - Where re-ranking fits in
- [[Hybrid Search]] - Initial retrieval stage before re-ranking
- [[Query Transformations]] - Query preprocessing (before retrieval)
- [[RAG Evaluation Metrics]] - Measuring re-ranking impact on pipeline quality
