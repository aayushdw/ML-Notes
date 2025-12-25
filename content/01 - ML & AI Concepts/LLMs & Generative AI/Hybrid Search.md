# Hybrid Search

## Overview

**Hybrid Search** is the gold standard for production RAG systems. It combines two fundamentally different retrieval approaches:
- **Sparse (Lexical)**: Fast, exact keyword matching (e.g., BM25)
- **Dense (Semantic)**: Neural embeddings that understand meaning and concepts

By running both methods and fusing their results, Hybrid Search achieves superior recall and relevance compared to using either method alone.

**Why it matters**: Most real-world applications benefit from this approach. Research shows Hybrid Search consistently outperforms single-method approaches in production scenarios.

---

## The Vocabulary Mismatch Problem (The Core Issue)

Hybrid Search exists to solve a fundamental problem in information retrieval:

### Problem 1: Sparse-Only Limitation
**Synonym/Paraphrase Mismatch**
- Query: "car"
- Relevant document: "Find an affordable **automobile**"
- Result: ❌ Missed! BM25 only matches exact keywords.

**Why it matters**: Users don't always use the exact terminology as the documents. A technical support bot searching for "fix the printer" shouldn't miss documents about "troubleshooting devices" or "resolving hardware issues."

### Problem 2: Dense-Only Limitation
**Domain-Specific Terms & Proper Nouns**
- Query: "XJ-900 specifications"
- Document A: "The **XJ-900** is our flagship product..."
- Document B: "This generic vehicle part is commonly used..."
- Result: ⚠️ Dense models struggle. They're trained on general text, not technical specifications or product codes.

**Why it matters**: In specialized domains (legal, medical, engineering), exact terminology is critical. A general embedding model may not understand that "ICD-10-CM" is more important than words like "the" or "and".

### The Hybrid Solution
Combine both strengths:
- **BM25** catches exact matches: "XJ-900", "ICD-10-CM", "SQL injection", etc.
- **Dense** catches concepts: "broken" ↔ "malfunctioning", "vehicle" ↔ "car", etc.
- **Together**: Complete coverage across vocabulary variations AND semantic understanding

---

## How Hybrid Search Works (High-Level Flow)

```
Query: "How do I fix a broken printer?"
       ↓
    ┌──┴──┐
    ↓     ↓
[BM25]  [Dense]
    ↓     ↓
List 1  List 2       (Two independent ranked lists)
    ↓     ↓
    └──┬──┘
       ↓
   [Fusion]          (RRF or Weighted Sum)
       ↓
   [Final Ranked List]  (Deduplicated, combined ranking)
       ↓
  Return Top-K Results
```

**Key insight**: BM25 and Dense retrieve different documents (partially overlapping). Fusion combines the evidence.

## The Two Pillars

### 1. Sparse Retrieval (Lexical)

**Method**: Inverted Index + [[BM25]] scoring (or TF-IDF)

**How it works**:
- Builds an inverted index: `word → list of documents containing that word`
- When you search, it finds all documents with exact keyword matches
- Ranks them using BM25 score (see [[BM25]] for detailed formula)

**Characteristics**:

| Aspect | Details |
|--------|---------|
| Speed | Very fast (simple index lookup) |
| Index Size | Small (just the inverted index) |
| Memory | Low (no neural models needed) |
| Training | None needed (works immediately) |
| Strengths | Exact matches, acronyms, proper nouns, product codes |
| Weaknesses | Misses synonyms and paraphrases |

**Examples of what it excels at**:
- Product ID searches: "SKU-12345-A"
- Acronyms: "XML", "API", "RAG"
- Named entities: "Microsoft", "COVID-19"
- Technical jargon: "ACID compliance", "normalization"

---

### 2. Dense Retrieval (Semantic)

**Method**: Bi-Encoders with neural embeddings (e.g., OpenAI `text-embedding-3`, Hugging Face `BGE-M3`)

**How it works**:
- Pre-trained neural network encodes text → dense vector (typically 384-1536 dimensions)
- Similarity = cosine distance between query vector and document vector
- Higher similarity = more semantically related

**Characteristics**:

| Aspect | Details |
|--------|---------|
| Speed | Fast (vector similarity is fast) |
| Index Size | Small (vectors are compact) |
| Memory | Medium (need to store model, indices) |
| Training | Pre-trained; can be fine-tuned for your domain |
| Strengths | Synonyms, paraphrases, cross-lingual, conceptual matching |
| Weaknesses | Domain shift (poor on unseen domains), struggles with exact tokens |

**Examples of what it excels at**:
- Paraphrases: "fix a flat" ↔ "tire repair"
- Synonyms: "automobile" ↔ "vehicle" ↔ "car"
- Intent matching: "how to learn Python" ↔ "Python tutorials"
- Cross-lingual: "Hello" ↔ "Hola" ↔ "你好"
- Concept drift: "broken car" ↔ "malfunctioning vehicle"

---

## Sparse vs Dense: Side-by-Side Comparison

| Query | Document | BM25 | Dense | Verdict |
|-------|----------|------|-------|---------|
| "Python tutorial" | "Learn Python programming" | ✅ Perfect match | ✅ Perfect match | Both find it |
| "How to fix a flat" | "Tire repair instructions" | ❌ No keyword overlap | ✅ Semantic match | Dense wins |
| "GPU performance" | "Graphics Processing Unit speed" | ❌ "GPU" ≠ "Graphics Processing Unit" | ⚠️ Some match | BM25 wins |
| "XJ-900 specs" | "The XJ-900 is a..." | ✅ Exact match | ⚠️ May miss (unfamiliar token) | BM25 wins |
| "vehicle" | "types of cars and trucks" | ❌ No exact match | ✅ Semantic match | Dense wins |

**Observation**: Each method misses cases the other catches. Hybrid Search combines them to avoid missing anything.

## Fusion Strategies

The core challenge: How do you combine results from two completely different scoring systems?
- **BM25 scores**: Range 0–40+ (unbounded, sparse)
- **Dense scores**: Range 0.0–1.0 (normalized, dense)

You need a fusion strategy. There are two main approaches:

---

### Strategy 1: Weighted Sum (Linear Combination)

**Idea**: Normalize both scores to 0-1 range, then take a weighted average.

$$\text{Final Score} = \alpha \cdot \text{Norm}(S_{dense}) + (1-\alpha) \cdot \text{Norm}(S_{sparse})$$

Where:
- $\alpha$ = weight for dense (0.0 to 1.0)
- $\text{Norm}(\cdot)$ = min-max normalization to [0, 1]
- $(1-\alpha)$ = weight for sparse

**Example**:
- BM25 raw score: 25 → Normalize to 0.85
- Dense raw score: 0.72 → Already normalized
- With α = 0.5 (50-50 split): Final = 0.5 × 0.72 + 0.5 × 0.85 = **0.785**

**Pros**:
- ✅ Intuitive (literally averaging the methods)
- ✅ Direct control over trade-off (tune α)
- ✅ Can give different weights to sparse vs dense

**Cons**:
- ❌ Requires score normalization (adds complexity)
- ❌ Sensitive to score distribution (changes per query type)
- ❌ Requires tuning α for your use case
- ❌ May need different α values for different domains

**When to use**:
- You have domain knowledge and want explicit control
- You can evaluate and tune α on your test set
- One method consistently outperforms the other in your domain

**Quick tips**:
- Start with α = 0.5 (50-50)
- Evaluate on your validation set
- Medical/legal domains: May need higher α (favor dense for domain shift protection)
- Product search: May need lower α (favor sparse for exact part numbers)

---

### Strategy 2: Reciprocal Rank Fusion (RRF) ⭐ **RECOMMENDED**

**Idea**: Don't use scores at all. Just use the **rank** (position) of each document in each retriever's result list.

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}(d, r)}$$

Where:
- $k$ = smoothing constant (usually 60)
- $\text{rank}(d, r)$ = position of doc $d$ in retriever $r$'s list (1st place = 1, 2nd place = 2, etc.)

**Example Calculation**:
- Doc A: Ranked #1 in BM25, #3 in Dense
  - RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01538 = **0.03177**
- Doc B: Ranked #5 in BM25, #1 in Dense
  - RRF = 1/(60+5) + 1/(60+1) = 0.01538 + 0.01639 = **0.03177**

Notice: Both get the same RRF score (≈0.032), but for different reasons!

**Intuition**:
- Being #1 in both lists = exponentially boosted
- Being #1 in one list, #50 in other = good (compound evidence)
- Being #100 in both lists = nearly irrelevant (too far down)

**Pros**:
- ✅ No score normalization needed
- ✅ Robust across all query types (no tuning required)
- ✅ No hyperparameter to tune (k=60 is standard)
- ✅ Industry standard (widely used in production)
- ✅ Handles score distribution differences automatically
- ✅ Simple to implement

**Cons**:
- ❌ Loses granular score information
- ❌ Less flexible if you want explicit weighting
- ❌ Less interpretable than weighted sum

**When to use**:
- **Default choice** for most production systems
- You don't have a validated test set to tune α
- You want robustness across diverse query types
- You want simplicity and reliability

**Why it works so well**:
RRF is theoretically sound (proven in information retrieval), practically simple, and empirically excellent. It's the industry standard because it "just works" across most scenarios without tuning.

---

## Comparison: Weighted Sum vs RRF

| Factor | Weighted Sum | RRF |
|--------|--------------|-----|
| **Tuning** | Requires α tuning | None (k=60 fixed) |
| **Score normalization** | Required | Not needed |
| **Complexity** | Medium | Simple |
| **Robustness** | Good (if α tuned) | Excellent (adaptive) |
| **Production readiness** | Good | ⭐ Best |
| **Interpretability** | High (explicit weights) | Medium (rank-based) |
| **When it shines** | Domain-specific optimization | General-purpose / unknown domains |

**Bottom line**: Start with RRF. Use Weighted Sum only if you can validate α on your data.

## Decision Guide: When to Use What

### Quick Decision Tree

```
Start here:
├─ "I need the absolute best relevance" → Hybrid (RRF)
├─ "I need extreme speed (sub-100ms)" → BM25 only
├─ "I have <1000 docs" → Dense only (simpler)
├─ "Unknown domain / cold start" → Hybrid (safest bet)
└─ "Specialized domain (medical/legal)" → Hybrid (catches domain terms)
```

### Detailed Scenarios

| Scenario | Best Choice | Why | Notes |
|----------|-------------|-----|-------|
| **Legal/Medical Documents** | **Hybrid** | Domain-specific terminology ("tort", "ICD-10-CM", "tort law") is critical. Dense alone may miss exact terms | Use Hybrid with RRF |
| **General Knowledge (Wikipedia)** | **Hybrid** | Mix of exact terms + synonyms | Perfect use case for Hybrid |
| **E-commerce Product Search** | **Hybrid** | Need both SKU matches (sparse) + semantic understanding (dense) | Higher α for exact part numbers |
| **Real-time Constraints (<100ms)** | **BM25** | Dense inference adds 50-200ms latency | Trade-off: less accuracy for speed |
| **Very Small Dataset (<1000 docs)** | **Dense** | BM25 overkill; Dense simpler to set up | Can use dense only |
| **Very Large Dataset (>10M docs)** | **Hybrid** | BM25 filters to top-1000, Dense reranks (two-stage) | Cost-efficient & accurate |
| **Multilingual Search** | **Dense** | Semantic models naturally handle cross-lingual | Can be Dense only |
| **Domain Shift Expected** | **Hybrid** | Dense weakens on new domains; BM25 is safety net | Critical for robustness |
| **Private/Sensitive Data** | **Hybrid** (BM25-heavy) | Dense requires external embeddings API | Use local embedding models or BM25 |
| **Unknown Domain (Cold Start)** | **Hybrid** | Most robust; handles any scenario | Default choice when unsure |

---

## Practical Implementation Patterns

### Pattern 1: Two-Stage Hybrid (Recommended for Scale)
```
Stage 1: BM25 retrieves top-1000 candidates (fast filter)
Stage 2: Dense reranks top-1000 (high quality)
Result: RRF fusion of both

Benefits: Speed + Quality
```

### Pattern 2: Parallel Hybrid (Simplest)
```
Run BM25 and Dense in parallel
Fuse results immediately
Return top-k

Benefits: Simplicity
```

### Pattern 3: Weighted Hybrid (Domain Optimized)
```
Run both in parallel
Weighted sum fusion (tuned α)
Return top-k

Benefits: Domain-specific optimization
```

---

## Benefits of Hybrid Search

### Completeness
- **No missed results**: Complementary strengths ensure high recall
- **Safety net**: If one method fails, the other catches it

### Robustness
- **Domain invariant**: Works across any domain
- **Query invariant**: Handles varied query styles
- **Degrades gracefully**: If dense embeddings are weak, BM25 compensates

### Practical
- **No tuning required** (with RRF): Works out-of-the-box
- **Interpretable**: Can see which method found what
- **Proven**: Used by Google, Pinecone, Elasticsearch, etc.

---

## Performance Metrics to Track

When evaluating Hybrid Search, measure:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Recall@k** | "Did we find the right doc in top-k?" | Higher is better |
| **NDCG@k** | "How well-ranked are the results?" | Higher is better |
| **MRR** | "How high is the first correct result?" | Higher is better |
| **Latency** | "How fast is retrieval?" | <100ms for interactive |
| **Cost** | "Embedding API calls, index size" | Lower is better |

**Pro tip**: Hybrid Search typically improves Recall@k and NDCG@k by 10-40% vs single methods.

## Advanced Architectures (Beyond Basic Hybrid)

These approaches try to solve limitations of basic Hybrid Search:

### 1. SPLADE (Sparse Lexical and Expansion)
**What if we could make sparse retrieval smarter?**

**Concept**: Learned Sparse Vectors that combine the interpretability of sparse search with the synonym-matching of dense search.

**How it works**:
- Uses a BERT model to learn **which terms to expand** a query with
- Outputs sparse vectors (non-zero values for relevant terms only)
- Uses inverted index just like BM25 (fast!)

**Example**:
- Input query: "car"
- Traditional sparse: Matches only docs with "car"
- SPLADE: Learns to expand with synonyms
- Output: `{"car": 1.0, "vehicle": 0.85, "automobile": 0.7, "motor": 0.65}`
- Result: Docs with "vehicle" are found even though they don't have "car"!

**Pros**:
- ✅ Combines sparse efficiency with semantic understanding
- ✅ Interpretable (can see which terms matched)
- ✅ No dense vectors needed (smaller index)

**Cons**:
- ❌ Requires SPLADE-specific indexing (not all databases support it)
- ❌ Less mature than basic Hybrid
- ❌ Training required

**When to use**: When you want semantic understanding WITHOUT the index overhead of dense vectors. Cutting-edge, not yet mainstream.

---

### 2. ColBERT (Late Interaction)
**What if we stored vectors for every token?**

**Concept**: Hybrid between bi-encoders (compress doc to 1 vector) and cross-encoders (full interaction).

**How it works**:
1. Encode document at **token-level** (not doc-level)
2. Store a vector for every token in the document
3. At query time, compute **MaxSim**: max similarity between each query token and document tokens
4. Sum MaxSim scores for final ranking

**Example**:
```
Query tokens: ["best", "AI", "paper"]

Doc: "This is the best AI research paper ever"
Doc tokens: [T1, T2, T3, T4, T5, T6, T7, T8]

For query token "best":     MaxSim = max(sim(best, T1), ..., sim(best, T8))
                            = sim(best, T4) = 0.99 (perfect match with T4="best")

For query token "AI":       MaxSim = max(sim(AI, T1), ..., sim(AI, T8))
                            = sim(AI, T5) = 0.98 (perfect match with T5="AI")

For query token "paper":    MaxSim = max(sim(paper, T1), ..., sim(paper, T8))
                            = sim(paper, T7) = 0.97 (perfect match with T7="paper")

Overall score: 0.99 + 0.98 + 0.97 = 2.94 (very high!)
```

**Pros**:
- ✅ SOTA (state-of-the-art) accuracy
- ✅ Fine-grained token-level matching
- ✅ Handles phrase matching naturally

**Cons**:
- ❌ Index size ~100x larger (vectors for every token!)
- ❌ Slower inference (more computation)
- ❌ Higher cost (storage + compute)

**When to use**: When accuracy is critical and budget permits (legal discovery, financial research, high-stakes applications). Not for real-time / cost-sensitive scenarios.

---

### 3. Understanding Domain Shift (Why Hybrid is Essential)

**The Problem**:
Dense embedding models are trained on general-purpose data:
- OpenAI embeddings: Trained on diverse internet text
- BGE-M3: Trained on web search & Wikipedia-like data

When you move to a specialized domain, accuracy often **drops sharply**.

**Real Examples**:
- Medical: "acute" = precise clinical term (not just "sharp")
- Legal: "consideration" = legal concept (not just "thinking about something")
- Finance: "yield" = investment return (not just "to give way")

**Why this happens**:
- Embedding model never learned domain-specific semantics
- Vector space doesn't distinguish domain-specific terms from generic ones

**Example Failure**:
```
Domain: Medical
Query: "acute myocardial infarction treatment"

Dense model (confused):
- "acute" is just "sharp" or "severe"
- "myocardial infarction" is unfamiliar tokens
- Returns generic medical articles instead of specific MI treatment docs

BM25 (works fine):
- "acute", "myocardial", "infarction", "treatment" = exact matches
- Returns relevant docs despite not understanding domain semantics
```

**The Solution**:
Hybrid Search + RRF ensures that even if dense fails, BM25 catches you. In domain-specific scenarios, BM25 often contributes 40-60% of the final ranking!

**Mitigation strategies**:
1. Use Hybrid Search with RRF (primary)
2. Fine-tune embedding model on domain data (if possible)
3. Use domain-specific embedding model (e.g., BioBERT for medical)
4. Increase BM25 weight (use Weighted Sum with α < 0.5)

## Implementation: LangChain EnsembleRetriever

### Simple RRF Hybrid (Recommended)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Pinecone

# 1. Initialize BM25 (sparse) retriever
bm25_retriever = BM25Retriever.from_documents(documents)

# 2. Initialize Dense (vector) retriever
vector_store = Pinecone(...)  # Any vector database
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 3. Create Ensemble with RRF (no tuning needed!)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    # No weights = RRF mode
)

# 4. Use it!
results = hybrid_retriever.invoke("How to fix a broken printer?")
# Returns: Top-10 deduplicated documents ranked by RRF
```

### Weighted Sum (If You Want to Tune)

```python
# Same as above, but specify weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.3, 0.7]  # 30% sparse, 70% dense
)
```

## Comprehensive Comparison Matrix

| Feature | BM25 | Dense | Hybrid (RRF) | SPLADE | ColBERT |
|---------|------|-------|--------------|--------|---------|
| **Recall@10** | ~60% | ~75% | **~85%** | ~80% | **~90%** |
| **Latency** | 5ms | 50ms | ~55ms | 5ms | 100ms |
| **Index Size** | 100MB | 500MB | 600MB | 200MB | 5GB |
| **Training Needed** | No | Pre-trained | No | Yes | Pre-trained |
| **Domain Shift** | Robust | ⚠️ Weak | ✅ Robust | ✅ Robust | ✅ Robust |
| **Exact Match** | ✅ Excellent | ⚠️ Poor | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Synonym Match** | ❌ Poor | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Production Ready** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Emerging | ⚠️ Expensive |
| **Setup Complexity** | Simple | Medium | Medium | Hard | Hard |
| **Cost to Run** | Low | Medium | Medium-High | Medium | High |

**Key insight**: Hybrid (RRF) provides the best balance of recall, robustness, and simplicity for most use cases.

---

## Learning Path

### Beginner
1. Understand the Vocabulary Mismatch Problem
2. Learn the basic difference: Sparse (exact) vs Dense (semantic)
3. Implement basic Hybrid with RRF

### Intermediate
1. Study BM25 formula (see [[BM25]] file)
2. Understand Weighted Sum fusion and when to use it
3. Experiment with different α values on your data
4. Evaluate using metrics like Recall@k and NDCG@k

### Advanced
1. Explore SPLADE for learned sparse representations
2. Implement ColBERT for high-accuracy scenarios
3. Fine-tune embedding models for your domain
4. Design two-stage retrieval (filter + rerank)

---

## Quick Checklist for Implementation

Before building Hybrid Search, ensure you have:

- [ ] **Documents**: Indexed and ready
- [ ] **Vector Database**: Set up (Pinecone, Weaviate, Milvus, etc.)
- [ ] **Embedding Model**: Chosen (OpenAI, Hugging Face, etc.)
- [ ] **BM25 Index**: Built (Elasticsearch, Lucene, etc.)
- [ ] **Fusion Strategy**: Decided (RRF recommended)
- [ ] **Test Set**: Created for evaluation
- [ ] **Metrics**: Decided (Recall@k, NDCG@k, MRR)
- [ ] **Baseline**: BM25-only results (to compare against)

---

## Common Questions Answered

### Q: Will Hybrid Search slow down my search?
**A**: Slightly. RRF adds ~10-20ms overhead (two parallel retrievals). Still <100ms total, acceptable for most applications.

### Q: Do I need to tune anything?
**A**: With RRF, no tuning required. With Weighted Sum, you need to tune α.

### Q: What if I can't store dense vectors due to space?
**A**: Try SPLADE (learned sparse representations) or use BM25 + re-ranking instead.

### Q: Do I need to fine-tune embeddings?
**A**: Only if you're in a specialized domain and have labeled data. For most cases, pre-trained embeddings are fine.

---

## Resources & References

### Foundational Papers
- **[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)**: Original RRF paper (Cormack et al., 2009)
- **[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)**: Standard information retrieval baseline
- **[SPLADE](https://arxiv.org/abs/2107.05720)**: Sparse Lexical and Expansion Model (Formal et al., 2021)
- **[ColBERT](https://arxiv.org/abs/2004.12832)**: Efficient and Effective Passage Search (Khattab & Zaharia, 2020)

### Blog Posts & Tutorials
- **[Pinecone: Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)**: Great practical introduction
- **[LangChain: Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)**: EnsembleRetriever documentation
- **[Elasticsearch: Hybrid Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-query.html)**: Production implementation

### Tools & Libraries
- **LangChain**: EnsembleRetriever (Python)
- **Elasticsearch**: Native hybrid search support
- **Pinecone**: Managed vector database with hybrid search
- **Milvus**: Open-source vector database
- **Weaviate**: Vector database with built-in hybrid

### Models
- **Dense Embeddings**: OpenAI `text-embedding-3`, Hugging Face `BGE-M3`, `BAAI/bge-large-en`
- **Fine-tuning**: Sentence Transformers, Hugging Face Transformers
- **BM25**: Elasticsearch, Lucene, Whoosh (Python)

---

## Personal Notes & Experiments
*   [Space for your thoughts and experiments...]

---

## Summary

**Hybrid Search is the industry standard** for production RAG systems because it combines the strengths of two fundamentally different retrieval approaches:

✅ **BM25 (Sparse)**: Fast, exact keyword matching, no training needed, robust across domains
✅ **Dense (Semantic)**: Understands synonyms, paraphrases, and concepts
✅ **RRF Fusion**: Simple, no tuning, robust across query types

**When to implement**:
- You want the best relevance (improve Recall by 10-40%)
- You're building a production RAG system
- You have diverse domains or unknown data
- You want safety against domain shift

**When to skip**:
- You need <50ms latency (use BM25 only)
- You have <1000 documents and no budget (use Dense only)
- You're doing keyword-only search (use BM25)

---

## Progress Checklist

- [ ] Understand the Vocabulary Mismatch Problem
- [ ] Compare Sparse (BM25) vs Dense (Semantic) retrieval
- [ ] Learn the difference between Weighted Sum and RRF
- [ ] Read [[BM25]] for detailed formula explanation
- [ ] Understand when to use BM25-only vs Dense-only
- [ ] Implement basic Hybrid Search with RRF
- [ ] Evaluate using Recall@k, NDCG@k, and MRR
- [ ] Explore advanced architectures (SPLADE, ColBERT) if needed
- [ ] Set up two-stage retrieval for large datasets

---

## See Also
- [[BM25]] - Detailed explanation of BM25 scoring formula
- [[RAG (Retrieval Augmented Generation) Overview]] - Back to RAG concepts
