# Query Transformations

## Overview

**Query Transformations** are preprocessing and expansion techniques applied to user queries *before* retrieval to improve the relevance of retrieved documents. Instead of using the user's raw question as-is, these techniques reframe, expand, or decompose the query to capture different angles, semantic variations, and implicit information.

**Core Insight**: A user's original query is often imperfect—it may be ambiguous, lack context, use different terminology than the indexed documents, or require multiple pieces of information to answer. Query transformations bridge this gap by generating alternative representations of the same intent.

### Why Query Transformations Matter

From the [[RAG (Retrieval Augmented Generation) Index#Phase 3 Advanced Retrieval|Advanced Retrieval]] phase:
- **Vocabulary Mismatch**: User asks "How do I fix my car's engine?" but documents use "automotive mechanics" or "mechanical repair"
- **Implicit Intent**: "I'm planning a trip" could need information on flights, hotels, travel guides, weather, visas, etc.
- **Context Loss**: Short queries like "Pricing" lack context about what product, service, or use case is being asked about
- **Semantic Variations**: Same question can be phrased many ways; single embedding may miss some phrasings

## Query Transformation Techniques

### 1. HyDE (Hypothetical Document Embeddings)

**Concept**: Hallucinate a hypothetical answer to the user's question, then use that answer's embedding as the search query instead of the original question's embedding.

**Why It Works**:
- The user's question is often a *query syntax* ("What is X?"), not a *document syntax* ("X is Y because...")
- By generating a plausible answer, we create a hypothetical document that shares vocabulary and semantic structure with real documents in the database
- Embedding space is optimized for document-to-document similarity, not question-to-document similarity

**Example**:
```
User Query: "What are the benefits of remote work?"

HyDE Step 1 - Generate hypothetical answer:
"Remote work offers flexible schedules, reduced commute time, and improved work-life balance.
Employees can save money on transportation and clothing. Companies reduce office space costs.
Productivity can increase due to fewer distractions. However, some report isolation and
communication challenges."

HyDE Step 2 - Embed the hypothetical answer (not the original question)

HyDE Step 3 - Retrieve documents similar to the hypothetical answer
→ Now you'll find documents about remote work benefits, employee testimonials, case studies, etc.
```

**Implementation**:
```python
# Pseudocode
def hyde_retrieval(user_query, llm, embedding_model, vector_store):
    # Step 1: Generate hypothetical document
    hypothetical_doc = llm.generate(
        prompt=f"Write a detailed answer to: {user_query}"
    )

    # Step 2: Embed hypothetical document
    query_embedding = embedding_model.embed(hypothetical_doc)

    # Step 3: Retrieve using the embedding
    results = vector_store.similarity_search(query_embedding, top_k=5)
    return results
```

**Strengths**:
- Simple and elegant—leverages LLM's ability to write natural documents
- Works well for factual, answer-seeking questions
- Single forward pass through embedding model

**Limitations**:
- Requires an extra LLM call (latency/cost trade-off)
- Can hallucinate plausible but incorrect details that mislead retrieval
- Less effective for questions without clear "correct answers" (e.g., subjective, creative queries)
- Performance depends on LLM quality and specificity of prompt

**Best For**:
- Factual Q&A ("What is...?", "When did...?", "How does...?")
- Knowledge-intensive tasks
- When latency is not critical

---

### 2. Multi-Query / Sub-Question Decomposition

**Concept**: Break a complex user query into multiple simpler sub-queries, retrieve results for each, then combine them.

**Why It Works**:
- Complex questions often have multiple "facets" or implicit sub-questions
- A single vector search may only capture one aspect; multiple searches cover different angles
- Different sub-queries may match different document clusters

**Example**:
```
User Query: "How do I set up a sustainable business while minimizing environmental impact?"

Multi-Query Decomposition:
1. "How to start a sustainable business"
2. "What are the best practices for environmental sustainability in business"
3. "Certification programs for eco-friendly companies"
4. "Cost-benefit analysis of sustainable vs traditional business practices"
5. "Supply chain optimization for environmental impact reduction"

→ Retrieve documents for each sub-query
→ Deduplicate and rank combined results
→ Pass all context to LLM for synthesis
```

**Implementation**:
```python
# Pseudocode
def multi_query_retrieval(user_query, llm, vector_store):
    # Step 1: Generate sub-queries
    prompt = f"""Generate 3-5 simple, independent sub-queries that capture
    all aspects of this complex question:
    "{user_query}"

    Format: One query per line"""

    sub_queries = llm.generate(prompt).split('\n')

    # Step 2: Retrieve for each sub-query
    all_results = []
    for sub_q in sub_queries:
        results = vector_store.similarity_search(sub_q, top_k=5)
        all_results.extend(results)

    # Step 3: Deduplicate and rank
    unique_results = deduplicate(all_results)
    ranked_results = rank_by_frequency(unique_results)

    return ranked_results
```

**Strengths**:
- Comprehensive coverage of complex questions
- Can catch information that wouldn't be found with a single query
- Each sub-query is simple and retrieval-friendly
- Works well with multi-hop reasoning scenarios

**Limitations**:
- Multiple LLM calls → higher latency and cost
- Can retrieve redundant or conflicting information
- Requires deduplication/ranking logic
- May over-fetch and increase context length

**Best For**:
- Complex, multi-faceted questions
- Research queries with multiple sub-topics
- Questions requiring synthesis across different domains
- When you can tolerate higher latency/cost

---

### 3. Query Expansion

**Concept**: Augment the original query with synonyms, related terms, and contextual variations while keeping it as a single search query.

**Why It Works**:
- Expands the "query vocabulary" to match more documents
- Helps with rare or specialized terminology
- Captures semantic cousins of the original terms

**Example**:
```
Original Query: "machine learning algorithms"

Expanded Query: "machine learning algorithms classification models supervised learning
AI pattern recognition neural networks statistical learning data science"

→ Single embedding captures broader semantic space
→ More likely to match documents using alternative terminology
```

**Implementation**:
```python
# Pseudocode
def query_expansion(user_query, llm, vector_store):
    # Step 1: Generate expanded terms
    prompt = f"""Given this query: "{user_query}"

    Generate related terms, synonyms, and variations (comma-separated):
    - Synonyms
    - Related concepts
    - Industry-specific terminology
    - Common variations
    """

    expanded_terms = llm.generate(prompt)
    expanded_query = user_query + " " + expanded_terms

    # Step 2: Single retrieval with expanded query
    results = vector_store.similarity_search(expanded_query, top_k=10)
    return results
```

**Strengths**:
- Single embedding call (vs multi-query's multiple calls)
- Simple to implement
- Good for terminology-heavy domains
- Balanced latency/coverage trade-off

**Limitations**:
- Can introduce noise by expanding too broadly
- May dilute the original query intent
- Works less well for semantic mismatches (more for vocabulary gaps)
- Embedding model may not weight all terms equally

**Best For**:
- Technical/specialized domains with specific terminology
- When documents use variant terminology
- Simple queries that need modest expansion

---

### 4. Query Rewriting / Clarification

**Concept**: Rewrite the user's query to be more explicit, removing ambiguity and adding implicit context.

**Why It Works**:
- Many queries are ambiguous or use pronouns/references that lack context
- Rewriting makes the query's intent explicit and document-like
- Helps with the question-vs-answer syntax mismatch

**Example**:
```
Original Query: "It's not working. How do I fix it?"

Rewritten Query: "My software application is not working properly.
What are the troubleshooting steps I should follow to identify and resolve the issue?"

→ More specific, self-contained, and likely to match technical documentation
```

**Implementation**:
```python
# Pseudocode
def query_rewriting(user_query, llm, vector_store):
    # Step 1: Rewrite for clarity
    prompt = f"""Rewrite this unclear query to be more explicit and self-contained.
    Add context, remove pronouns/references, and make the intent crystal clear:

    "{user_query}"

    Rewritten query:"""

    rewritten = llm.generate(prompt)

    # Step 2: Retrieve using rewritten query
    results = vector_store.similarity_search(rewritten, top_k=10)
    return results
```

**Strengths**:
- Handles ambiguous and underspecified queries
- Single retrieval call
- Particularly useful for conversational interfaces
- Can improve context relevance significantly

**Limitations**:
- Requires high-quality LLM for good rewriting
- Can lose nuance if over-generalized
- May not help with true vocabulary gaps
- Adds latency

**Best For**:
- Conversational Q&A systems
- User queries that are vague or context-dependent
- When query clarification is the main issue

---

### 5. Query Routing

**Concept**: Use a classifier to route queries to different retrieval strategies based on query type.

**Why It Works**:
- Different query types benefit from different retrieval approaches
- Classification is fast and cost-effective
- Allows optimization of specific retrieval paths

**Example**:
```
User Query: "What's the price of your premium plan?"

Router Classification: METADATA_QUERY → Route to metadata filter search

---

User Query: "Why should I choose your product over competitors?"

Router Classification: SEMANTIC_QUERY → Route to hybrid search or re-ranking

---

User Query: "How do I integrate your API into my application?"

Router Classification: MULTI_HOP_QUERY → Route to multi-query or step-back prompting
```

**Implementation**:
```python
# Pseudocode
def query_routing(user_query, classifier):
    # Step 1: Classify query type
    query_type = classifier.predict(user_query)
    # Returns: "METADATA", "SEMANTIC", "MULTI_HOP", "COMPARISON", etc.

    # Step 2: Route to appropriate retrieval strategy
    if query_type == "METADATA":
        results = metadata_filtered_search(user_query)
    elif query_type == "MULTI_HOP":
        results = multi_query_retrieval(user_query)
    elif query_type == "COMPARISON":
        results = comparative_search(user_query)
    else:
        results = standard_hybrid_search(user_query)

    return results
```

**Strengths**:
- Efficient: only uses expensive techniques when needed
- Optimizable: can fine-tune each path separately
- Scalable: doesn't degrade with more query types
- Reduces hallucination by using appropriate strategies

**Limitations**:
- Requires training/fine-tuning a classifier
- Misclassification can degrade performance
- Adds complexity to the system
- Cold-start problem with new query types

**Best For**:
- Production systems with diverse query types
- When you can afford classifier development
- Cost-optimized systems where not all queries need expensive processing

---

### 6. Step-Back Prompting

**Concept**: Ask the LLM to abstract away from specific details and think about the broader concepts/principles before retrieving.

**Why It Works**:
- High-level concepts often have better document representation than specific details
- Helps with bridging vocabulary gaps between user-specific context and document corpus
- Captures fundamental principles that apply broadly

**Example**:
```
Specific Query: "How do I debug a NaN error in my PyTorch model's loss calculation?"

Step-Back Query: "What are the fundamental principles of debugging machine learning models
and troubleshooting mathematical operations in neural networks?"

→ Retrieves documents about:
  - General ML debugging practices
  - Numerical stability in deep learning
  - PyTorch best practices
  - Common ML pitfalls
→ More useful than searching specifically for "NaN errors"
```

**Implementation**:
```python
# Pseudocode
def step_back_prompting(user_query, llm, vector_store):
    # Step 1: Generate high-level abstract query
    prompt = f"""Given this specific technical question:
    "{user_query}"

    What are the broader concepts, principles, or categories this question belongs to?
    Generate a more abstract, fundamental question that captures the core issue:"""

    abstract_query = llm.generate(prompt)

    # Step 2: Retrieve using both original and abstract
    specific_results = vector_store.similarity_search(user_query, top_k=5)
    abstract_results = vector_store.similarity_search(abstract_query, top_k=5)

    # Step 3: Combine and deduplicate
    combined = deduplicate(specific_results + abstract_results)
    return combined
```

**Strengths**:
- Bridges vocabulary and abstraction gaps
- Effective for finding foundational knowledge
- Can be combined with other techniques
- Works well with principled/conceptual documents

**Limitations**:
- Can lose important specifics
- Requires good LLM for abstraction
- May retrieve overly general documents
- Two retrieval calls increases cost/latency

**Best For**:
- Technical/educational domains
- When specific terminology isn't in documents
- Questions about underlying concepts/principles

---

### 7. Perspective-Based Query Generation

**Concept**: Generate multiple perspectives or "personas" asking the same question to capture different vocabulary and phrasings.

**Why It Works**:
- Different people describe the same problem differently
- Different expertise levels use different terminology
- Captures vocabulary from multiple viewpoints

**Example**:
```
Original Question: "machine learning"

Beginner Perspective: "How do I get started with machine learning as a complete beginner?"

Researcher Perspective: "What are the latest advances in deep learning and neural network architectures?"

Practitioner Perspective: "How do I deploy machine learning models to production and maintain them?"

Industry Perspective: "What are the business applications and ROI of machine learning?"

→ Retrieve with each perspective to cover different document clusters
```

**Implementation**:
```python
# Pseudocode
def perspective_based_retrieval(user_query, llm, vector_store):
    perspectives = ["beginner", "researcher", "practitioner", "industry_expert"]

    all_results = []
    for perspective in perspectives:
        prompt = f"""Rephrase this question from a {perspective} perspective:
        "{user_query}"

        Focus on what a {perspective} would ask:"""

        rephrased = llm.generate(prompt)
        results = vector_store.similarity_search(rephrased, top_k=3)
        all_results.extend(results)

    combined = deduplicate(all_results)
    return combined
```

**Strengths**:
- Comprehensive coverage from multiple viewpoints
- Captures diverse vocabulary and phrasings
- Especially useful for broad topics
- Naturally handles different expertise levels

**Limitations**:
- Multiple LLM calls (latency/cost)
- Can be overkill for narrow queries
- Requires deduplication and ranking
- Risk of retrieving irrelevant perspective-specific content

**Best For**:
- Broad educational/reference topics
- When documents are written for diverse audiences
- Comprehensive Q&A systems

---

## Comparison Table

| Technique | Latency | Cost | Query Complexity | Best For | Key Advantage | Main Trade-off |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HyDE** | 1-2 calls | Low | Factual, answer-seeking | Factual Q&A | Elegant, works well for facts | Can hallucinate details |
| **Multi-Query** | 3-5 calls | Medium | Multi-faceted, complex | Complex questions | Comprehensive coverage | Redundancy, context bloat |
| **Query Expansion** | 1 call | Low | Terminology gap | Domain-specific searches | Balanced coverage | Noise introduction |
| **Query Rewriting** | 1 call | Low | Ambiguous, vague | Conversational Q&A | Handles ambiguity | May lose nuance |
| **Query Routing** | 1 call | Very Low | Diverse types | Production systems | Efficiency | Requires classifier |
| **Step-Back** | 2 calls | Medium | Concept-based | Principles/foundations | Bridges abstraction | Loses specificity |
| **Perspective-Based** | 4+ calls | High | Broad topics | Educational QA | Multi-viewpoint coverage | Expensive, overkill for narrow Q |

---

## Integration with the RAG Pipeline

Query transformations sit at the boundary between **query processing** and **retrieval**:

```mermaid
graph LR
    User["User Query"] --> Transform["Query Transformation"]
    Transform --> |Expanded/Refined Query| Hybrid["Hybrid Search"]
    Hybrid --> |Retrieved Chunks| Rerank["Re-ranker"]
    Rerank --> |Top-K Results| LLM["LLM Generator"]
    LLM --> Answer["Final Answer"]
```

### Connection to Advanced Retrieval Phase
As noted in the [[RAG (Retrieval Augmented Generation) Index#Phase 3 Advanced Retrieval|RAG Overview]], query transformations are part of improving signal-to-noise ratio:

- **Before**: Raw user query → Embedding → Retrieval
  - Problem: Question syntax doesn't match document syntax
  - Problem: Ambiguity and incomplete context

- **After**: Raw user query → **Transform** → Document-like representation → Embedding → Retrieval
  - Benefit: Better matching with document corpus
  - Benefit: Captures multiple angles and implicit information

---

## Implementation Patterns

### Sequential Transformation
```python
# Apply one transformation after another
def sequential_transform(query, transformers):
    result = query
    for transformer in transformers:
        result = transformer(result)
    return result

# Example: Rewrite → Multi-Query → Expansion
```

### Ensemble Transformation
```python
# Apply multiple transformations in parallel, retrieve for each, combine results
def ensemble_transform(query, transformers):
    results = []
    for transformer in transformers:
        transformed = transformer(query)
        retrieved = retriever.search(transformed)
        results.extend(retrieved)
    return deduplicate_and_rank(results)

# Example: HyDE + Multi-Query + Step-Back in parallel
```

### Adaptive Transformation
```python
# Choose transformations based on query characteristics
def adaptive_transform(query):
    if len(query) < 10:  # Very short query
        return query_expansion(query)
    elif has_ambiguity(query):  # Ambiguous/vague
        return query_rewriting(query)
    elif is_complex(query):  # Multi-faceted
        return multi_query_decomposition(query)
    else:
        return query  # No transformation needed
```

---

## Trade-offs & Decision Framework

### When to Use Query Transformations

**Use Minimal/No Transformation**:
- ✓ Short, specific, well-defined queries
- ✓ Queries with clear domain terminology
- ✓ When latency is critical
- ✗ Will miss context and variations

**Use Single Transformation (HyDE / Rewriting / Expansion)**:
- ✓ Moderate complexity, single main issue
- ✓ Balanced latency/quality requirement
- ✓ Production systems with cost constraints
- ✗ May miss multiple perspectives

**Use Multiple Transformations (Ensemble)**:
- ✓ Complex, multi-faceted questions
- ✓ Research/comprehensive answers required
- ✓ When latency and cost are acceptable
- ✗ Higher operational complexity

**Use Query Routing**:
- ✓ Diverse query types at scale
- ✓ Production system optimization needed
- ✓ Different query types have very different characteristics
- ✗ Requires classifier development and maintenance

---

## Practical Considerations

### Prompt Engineering for Transformations
- **Be Specific**: "Generate 3-5 independent sub-queries" (better) vs "Break down the query" (vague)
- **Show Examples**: Few-shot prompts improve LLM behavior
- **Clarify Output Format**: "One per line", "JSON format", etc.
- **Provide Context**: Sometimes include expected document types or domains

### Deduplication & Ranking
- **Semantic Deduplication**: Use embedding distance or cosine similarity to find near-duplicates
- **Frequency-Based Ranking**: Documents retrieved multiple times are likely more relevant
- **Diversity**: Sometimes want to maintain different perspectives rather than deduplicate everything

### Cost Optimization
- **Caching**: Store common transformations and their results
- **Batching**: When using ensemble approaches, batch LLM calls when possible
- **Selective Application**: Use routing or heuristics to apply expensive transformations only when needed

---

## Limitations & When Query Transformations Can Hurt

- **Over-Transformation**: Too many transformations → context bloat, contradictions, noise
- **Hallucination Amplification**: Each LLM call risks introducing hallucinations that guide retrieval
- **Semantic Drift**: Transformations can inadvertently change query meaning
- **Vocabulary Explosion**: Query expansion on already-comprehensive queries just adds noise

**Best Practice**: Start simple, measure retrieval quality metrics ([[RAG Evaluation Metrics]]), and add transformations only where needed.

---

## See Also
- [[RAG (Retrieval Augmented Generation) Index#Phase 3 Advanced Retrieval|Advanced Retrieval Phase]] - Where query transformations fit
- [[Hybrid Search]] - Often used with query transformations
- [[Re-ranking]] - Post-retrieval refinement (complements query transformation)
- [[RAG Evaluation Metrics]] - Measuring retrieval quality after transformations

---

## Personal Notes
*[Space for your thoughts...]*

## Progress Checklist
- [ ] Understand basic query transformation concepts
- [ ] Learn 2-3 specific techniques (HyDE, Multi-Query, etc.)
- [ ] Understand trade-offs between techniques
- [ ] Implement one technique in a project
- [ ] Measure impact on retrieval quality
- [ ] Can explain to others

**Back to**: [[01 - ML & AI Concepts/Index]]
