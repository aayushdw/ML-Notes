## Overview

**Re-ranking** is a post-retrieval refinement step that re-scores and reorders the top-K candidates retrieved by the initial retrieval system. Instead of relying solely on the first retrieval pass (which uses approximate/fast similarity metrics), re-ranking applies a more sophisticated, expensive model to re-evaluate and reorder results for higher precision.

**Core Insight**: The initial retriever is optimized for *recall* (finding all potentially relevant documents quickly), often using fast approximate methods. Re-ranking is optimized for *precision* (finding the most relevant documents from the candidates). This two-stage approach balances speed and accuracy.

### Why Re-ranking Matters

From the [[01 - RAG Index#Phase 3 Advanced Retrieval|Advanced Retrieval]] phase:
- **Imperfect Initial Retrieval**: Dense retrievers (vector search) and sparse retrievers (BM25) often rank relevant documents below irrelevant ones
- **Lost in the Middle**: More relevant documents may be ranked 3rd or 4th, while the LLM uses only the top results; re-ranking ensures truly relevant docs appear first
- **Noise Reduction**: Retrieved results often include false positives; re-ranking filters these out
- **Quality Improvement**: Using a more sophisticated model for ranking can significantly improve downstream answer quality

### The Two-Stage Retrieval Paradigm

```
Stage 1 (Retrieval): Fast, Approximate, High Recall
├─ Sparse (BM25): Keyword matching
├─ Dense (Vector Search): Semantic similarity
└─ Hybrid: Combination of both

    ↓ (returns ~100 candidates)

Stage 2 (Re-ranking): Slow, Exact, High Precision
└─ Cross-Encoder: Fine-grained relevance scoring

    ↓ (returns reordered top-10)

Final Context for LLM
```

---

## Core Concept: Cross-Encoder vs Bi-Encoder

### Bi-Encoder (Dual-Encoder) - Initial Retrieval

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

---

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

---

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

---

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
- When you need interpretable relevance scores
- Filtering and thresholding decisions

---

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

**Implementation**:
```python
# Pseudocode - LambdaMART or RankNet style
def pairwise_rerank(query, candidates, pairwise_model):
    # Model trained to predict: which of two docs is more relevant?
    scores = []

    for i, doc1 in enumerate(candidates):
        for j, doc2 in enumerate(candidates[i+1:]):
            # Model outputs: 1 if doc1 more relevant, 0 if doc2 more relevant
            preference = pairwise_model.score_pair(query, doc1, doc2)
            if preference == 1:
                # doc1 wins this comparison
                pass

    # Aggregate pairwise preferences into total order
    reranked = compute_total_order(preferences)
    return reranked
```

**Strengths**:
- Can be more accurate than pointwise (uses comparison information)
- Better at handling ranking tasks where relative order matters most
- Used in learning-to-rank (LambdaMART, RankNet)

**Limitations**:
- Requires pairwise training data (expensive to create)
- Doesn't provide absolute relevance scores
- O(n²) comparisons for n documents

**Best For**:
- Large-scale ranking systems (search engines)
- When you need perfect ordering, not just relevance scores
- Systems with access to pairwise training data

---

## Re-ranking Models & Strategies

### 1. Cross-Encoder Models

**Pre-trained Models** (fine-tuned on MS MARCO, Natural Questions, etc.):
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: Fast, good accuracy
- `cross-encoder/mmarco-MiniLMv2-L12-H384-v1`: Multilingual
- `cross-encoder/qnli-distilroberta-base`: Lightweight
- `cross-encoder/ms-marco-TinyBERT-L-2-v2`: Ultra-fast, lower accuracy

**Cost-Accuracy Trade-off**:
```
Accuracy ↑
     ^
     |  Large BERT
     |   •
     |      •  MiniLM
     |         •   DistilBERT
     |            •
     |   TinyBERT    •
     |      •
     +──────────────────→ Latency
     Latency ↑
```

---

### 2. Hybrid Re-ranking

**Concept**: Combine multiple re-ranking signals (cross-encoder score + BM25 + metadata filters + LLM judgment).

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
- When metadata is available and meaningful
- Systems with different document types

---

## Integration with the RAG Pipeline

Re-ranking appears at a critical juncture, after initial retrieval but before LLM generation:

```
┌─────────────────────────────────────────────────────┐
│ User Query                                          │
└──────────────────────┬────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ Query Transformation (Optional)                     │
│ [[Query Transformations]] - HyDE, Multi-Query, etc. │
└──────────────────────┬────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ Stage 1: Initial Retrieval (Fast, High Recall)     │
│ [[Hybrid Search]] - BM25 + Dense                   │
│ ↓                                                   │
│ Returns ~100 candidates                             │
└──────────────────────┬────────────────────────────┘
                       ↓
        ╔══════════════════════════════════╗
        ║  Stage 2: RE-RANKING (THIS PAGE) ║  ← YOU ARE HERE
        ║  Cross-Encoder Re-scoring        ║
        ║  ↓                               ║
        ║  Returns reordered top-K         ║
        ╚══════════════════════════════════╝
                       ↓
┌─────────────────────────────────────────────────────┐
│ Context Assembly                                    │
│ Chunk & format top-K for prompt                    │
└──────────────────────┬────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ LLM Generation                                      │
│ Answer question using reranked context             │
└──────────────────────┬────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ Final Answer                                        │
└─────────────────────────────────────────────────────┘
```

### Why This Ordering Matters

1. **Query Transformation** → Makes query better match document vocabulary
2. **Initial Retrieval** → Casts a wide net, prioritizes recall
3. **Re-ranking** → Filters and reorders for precision (THIS STAGE)
4. **LLM** → Generates using the best documents

Without re-ranking, the LLM works with whatever the retriever found, which may include noise or miss relevant documents ranked 11-100.

---

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

**Decision Matrix**:
```
             Latency Critical    Balanced         Accuracy Critical
             (< 100ms)          (< 500ms)         (No limit)

No Reranking    ✓ Good          ✗ Poor           ✗ Poor
Fast Model      ✗ Slow          ✓ Good           ✓ Good
Large Model     ✗ Too Slow      ✗ Too Slow       ✓ Excellent
```

---

## Implementation Patterns

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

---

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

---

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

This pattern balances cost (only top 20 see the slow model) and accuracy.

---

## Evaluation & Metrics

### Key Metrics for Re-ranking

**NDCG@K (Normalized Discounted Cumulative Gain)**:
- Measure: How good is the ranking compared to ideal ranking?
- Formula: `NDCG = DCG / IDCG` where DCG penalizes lower-ranked relevant docs
- Range: 0-1 (1 is perfect ranking)
- Use: Standard for ranking quality

**MRR@K (Mean Reciprocal Rank)**:
- Measure: On average, how early is the first relevant result?
- Formula: `MRR = 1/rank_of_first_relevant`
- Best for: Single-answer scenarios (FAQ, entity lookup)

**MAP@K (Mean Average Precision)**:
- Measure: Precision at each relevant document
- Good for: Evaluating comprehensive answer retrieval

**Recall@K**:
- Measure: Out of all relevant documents, what % did we retrieve?
- Formula: `relevant_retrieved / total_relevant`

**Example Evaluation**:
```
Query: "How to fix a NaN error in PyTorch?"

Ideal ranking (by humans):
1. "Debugging NaN in PyTorch Loss" [Relevance: 2/2]
2. "Numerical Stability in Neural Nets" [Relevance: 2/2]
3. "PyTorch Debugging Best Practices" [Relevance: 1/2]

Before reranking:
1. "PyTorch Documentation" [Score: 0.85] [Relevance: 1/2]
2. "Debugging NaN in PyTorch Loss" [Score: 0.78] [Relevance: 2/2]
3. "Related: GPU Memory Issues" [Score: 0.75] [Relevance: 0/2]

NDCG@3 before = 0.72

After reranking (with cross-encoder):
1. "Debugging NaN in PyTorch Loss" [Score: 0.92] [Relevance: 2/2]
2. "Numerical Stability in Neural Nets" [Score: 0.88] [Relevance: 2/2]
3. "PyTorch Debugging Best Practices" [Score: 0.81] [Relevance: 1/2]

NDCG@3 after = 0.98

Improvement: +26%
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Over-Relying on Re-ranking

**Problem**: Using re-ranking to compensate for poor initial retrieval.
- If retrieval gets only 50% recall, re-ranking can't fix this (can't rerank what wasn't retrieved)

**Solution**:
- First optimize initial retrieval ([[Hybrid Search]], [[Query Transformations]])
- Use re-ranking as refinement, not rescue

---

### Pitfall 2: Model-Data Mismatch

**Problem**: Using cross-encoder trained on MS MARCO for medical documents.
- Mismatch between training data and target domain

**Solution**:
- Fine-tune on domain-specific data if possible
- Use small cross-encoder (faster to fine-tune)
- Fall back to ensemble methods if fine-tuning unavailable

---

### Pitfall 3: Computational Overhead

**Problem**: Re-ranking 1000 candidates with large BERT model = 10+ seconds.
- Defeats the purpose of fast retrieval

**Solution**:
- Multi-stage: Fast model on 100, then slow model on 20
- Batch processing to utilize GPU
- Use lighter models (MiniLM, TinyBERT)
- Selective re-ranking: only rerank top-K from initial retrieval

---

### Pitfall 4: Score Miscalibration

**Problem**: Cross-encoder outputs [0.92, 0.78, 0.45] but you don't know what these mean absolutely (just relative).
- Can't use raw scores for filtering/thresholding

**Solution**:
- Use models trained with proper calibration (temperature scaling)
- Don't rely on absolute score values; use relative ranking
- Test thresholds empirically on your data

---

## Comparison with Alternatives

| Approach | Latency | Accuracy | Complexity | Cost | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No Re-ranking** | 50ms | Low | 0 | $0 | Real-time, low stakes |
| **Cross-Encoder (Fast)** | 200ms | Medium | 1 | Low | Balanced systems |
| **Cross-Encoder (Large)** | 1-2s | High | 1 | Medium | Accuracy-critical |
| **LLM Re-ranking** | 2-5s | Very High | 2 | High | Complex relevance |
| **Hybrid** | 300-500ms | High | 3 | Medium | Production systems |
| **Learning-to-Rank** | 100-300ms | Very High | 4 | High | Search engines |

---

## Integration with Query Transformations and Hybrid Search

**Full Advanced Retrieval Pipeline**:

```
User Query
    ↓
[Query Transformations] [[Query Transformations]]
├─ HyDE: Hypothetical answer
├─ Multi-Query: Decompose into sub-queries
└─ Query Expansion: Add synonyms
    ↓ (Multiple refined queries)
[Hybrid Search] [[Hybrid Search]]
├─ Sparse (BM25): Keyword match
├─ Dense (Vector): Semantic match
└─ Combine: Union or weighted sum
    ↓ (100+ candidates)
[RE-RANKING] (THIS PAGE)
├─ Cross-Encoder: Fine-grained scoring
├─ Pointwise: Score each doc independently
└─ Reorder: Put best docs first
    ↓ (10 top candidates)
[LLM Generation]
├─ Context assembly
├─ Prompt engineering
└─ Answer generation
    ↓
Final Answer
```

**Key Synergies**:
1. **Query Transformation** makes retrieval catch more relevant docs
2. **Hybrid Search** gets broad coverage (recall)
3. **Re-ranking** filters and orders for precision
4. Together → best docs with high probability end up in top-K

---

## See Also
- [[01 - RAG Index#Phase 3 Advanced Retrieval|Advanced Retrieval Phase]] - Where re-ranking fits in
- [[Hybrid Search]] - Initial retrieval stage before re-ranking
- [[Query Transformations]] - Query preprocessing (before retrieval)
- [[RAG Evaluation Metrics]] - Measuring re-ranking impact on pipeline quality
