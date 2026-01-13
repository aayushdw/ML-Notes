## Overview
**GraphRAG** is an advanced Retrieval-Augmented Generation approach that augments traditional RAG with knowledge graph structures and hierarchical community detection. Instead of retrieving isolated text chunks based on vector similarity, GraphRAG extracts entities and their relationships from documents to build a structured knowledge graph, then uses community detection algorithms to create hierarchical summaries of the entire corpus.

The key innovation is that GraphRAG can answer **global queries** (questions about the entire corpus, like "What are the main themes in this dataset?") that traditional vector-based RAG completely fails at, while also enhancing **local queries** with relationship context that would be lost in chunked text.

## Problem with Standard RAG
Standard [[Naive RAG Pipeline|Naive RAG]] retrieves chunks purely based on semantic similarity to the query. This approach has a fundamental blind spot.

For a query like "What are the main themes across all documents?", there is no single chunk that contains this answer. The query itself is not semantically similar to any specific chunk, so standard RAG returns low-relevance results.

## How GraphRAG Solves This

GraphRAG introduces a two-phase approach:

**Phase 1: Indexing (Offline Pre-computation)**
1. Extract entities (people, places, concepts) and relationships from all documents using an LLM
2. Build a **knowledge graph** where nodes = entities, edges = relationships
3. Apply community detection ([Leiden algorithm](https://en.wikipedia.org/wiki/Leiden_algorithm)) to cluster related entities
4. Pre-generate **summaries for each community** at multiple hierarchy levels

**Phase 2: Query Time**
- For **local queries**: Traverse entity neighborhoods in the graph
- For **global queries**: Map-reduce over community summaries


![[GraphRAG 2026-01-10 17.51.20.excalidraw.svg]]

### Community Detection

The ([Leiden algorithm](https://en.wikipedia.org/wiki/Leiden_algorithm)) partitions the knowledge graph into "communities" of densely connected entities. 
The communities are hierarchical. 
Note that each entity belongs to a single leaf-level community.

![[GraphRAG 2026-01-10 17.59.44.excalidraw.svg]]


## Architecture: The Full Pipeline

### 1. Indexing Phase (Expensive, Upfront)

```
Source Documents
      │
      ▼ (1) Chunk Documents
Text Chunks (~600 tokens, with overlap)
      │
      ▼ (2) LLM Entity & Relationship Extraction
Entities: [(name, type, description), ...]
Relationships: [(source, target, description), ...]
      │
      ▼ (3) Entity Resolution
Canonical Nodes (deduplicated entities)
      │
      ▼ (4) Build Knowledge Graph
Graph G = (V, E) where V=entities, E=relationships
      │
      ▼ (5) Community Detection (Leiden Algorithm)
Hierarchical Community Structure
      │
      ▼ (6) LLM Community Summarization
Community Summaries at each hierarchy level
      │
      ▼ (7) Embed Everything
Vector indices for entities, communities, relationships
```

#### Step 2: Entity Extraction Detail
GraphRAG uses multipart prompts to guide the LLM:

```
Prompt Structure:
1. "Identify all entities in this text (people, organizations, places, concepts)"
2. "For each entity, provide: name, type, and a brief description"
3. "Identify all relationships between entities"
4. "For each relationship, describe the connection"

Gleaning (Multi-pass extraction):
After initial extraction, follow-up prompt:
"Review the text again. Did you miss any entities or relationships?"
→ Significantly reduces information loss
```

#### How Does This Work If the Corpus Exceeds Context Window?

A natural question: if we need RAG because the corpus is too large for the LLM context window, how can the LLM extract entities and relationships from the entire corpus?

**GraphRAG processes chunks individually during indexing, then merges the extracted knowledge.**

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Document ──► Chunk 1 ──► LLM ──► Entities/Relations       │
│             ──► Chunk 2 ──► LLM ──► Entities/Relations       │
│             ──► Chunk 3 ──► LLM ──► Entities/Relations       │
│                    ...                                       │
│                    ↓                                         │
│            ┌──────────────────┐                              │
│            │  Merge & Dedupe  │                              │
│            │ (Entity Resolution)                             │
│            └────────┬─────────┘                              │
│                     ↓                                        │
│            Unified Knowledge Graph                           │
└─────────────────────────────────────────────────────────────┘
```

**Handling cross-chunk relationships:**

1. **Entity Resolution**: "Steve Jobs" in chunk 1 and "Jobs" in chunk 47 are resolved to the same node (via embedding similarity or LLM-based matching)

2. **Gleaning across context**: Some implementations show entity lists from neighboring chunks during extraction to catch missed connections

3. **Graph structure completes the picture**: Once entities exist as nodes, relationships spanning distant parts of the corpus are connected transitively. If A→B appears in chunk 1 and B→C in chunk 50, the graph still connects A to C through B

**The trade-off**: GraphRAG captures relationships *within* a chunk very well, but relationships spanning distant parts of the corpus require either:
- The entities appearing together in at least one chunk
- Explicit entity resolution/linking during merge
- Community detection grouping related entities post-hoc

This is why [[Chunking Strategies|chunking strategy]] matters even more in GraphRAG: you want semantically coherent chunks where related entities co-occur.


### 2. Query Phase

GraphRAG supports two distinct query modes:

#### Local Search (Entity-Centric)
For questions like: "Who founded SpaceX?"

```
Query → Extract Key Entities → Traverse Graph Neighborhood → Return Subgraph
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │        SpaceX                   │
                    │       /      \                  │
                    │  Elon Musk   Starship          │
                    │      │         │               │
                    │   Tesla    Mars Missions       │
                    └─────────────────────────────────┘
```

**Process**:
1. Use the LLM to extract entities from the query
2. Find those entities in the knowledge graph
3. Expand to neighboring nodes (1-2 hops)
4. Retrieve associated text chunks and relationship descriptions
5. Generate answer from retrieved context

**Characteristics**: Fast, targeted, lower cost per query.

#### Global Search (Corpus-Wide)
For questions like: "What are the main themes in this dataset?"

```
Query → Identify Relevant Community Level → Map-Reduce Over Summaries

Level Selection:
  Level 0 (1 community)   → Coarsest (entire corpus)
  Level 1 (~10 communities) → Broad themes
  Level 2 (~50 communities) → Sub-topics
  ...

Map-Reduce:
  Map: Send query + each community summary to LLM → Partial answers
  Reduce: Synthesize partial answers into final response
```

**Process**:
1. Select appropriate hierarchy level based on query specificity
2. For each community at that level, ask LLM: "Based on this summary, what's relevant to the query?"
3. Collect all partial answers
4. Final LLM call: Synthesize partial answers into coherent response

**Characteristics**: Higher latency (10+ LLM calls), expensive (200K+ tokens), but answers previously impossible queries.

## Mathematical Foundation

### Graph Representation
Given a corpus $D = \{d_1, d_2, ..., d_n\}$, GraphRAG constructs:

- **Entity set**: $V = \{v_1, v_2, ..., v_m\}$ where each $v_i = (name, type, description)$
- **Relationship set**: $E = \{(v_i, v_j, r) : v_i, v_j \in V, r = relationship\_description\}$
- **Knowledge Graph**: $G = (V, E)$

### Community Hierarchy
After Leiden community detection:

$$C = \{C_0, C_1, ..., C_k\}$$

Where $C_l = \{c_{l,1}, c_{l,2}, ..., c_{l,j}\}$ represents communities at level $l$, and:
- $C_0$ = single community (entire graph)
- $C_k$ = finest communities (most granular)
- $|C_l| < |C_{l+1}|$ (more communities at finer levels)

For each community $c \in C_l$:
$$summary(c) = LLM(\text{entities in } c, \text{relationships in } c)$$

### Query Processing

**Local Search** retrieves a subgraph:
$$G_{local}(q) = \{v \in V : distance(v, entities(q)) \leq k\}$$

**Global Search** uses map-reduce:
$$answer(q) = Reduce(Map(q, \{summary(c) : c \in C_l\}))$$

## Practical Application

### When to Use GraphRAG

| Use Case | Why GraphRAG Helps |
|:---------|:-------------------|
| **Thematic analysis** | "What are the main topics in these 1000 documents?" |
| **Relationship reasoning** | "How does Company A connect to Person B?" |
| **Multi-hop questions** | Chain of relationships naturally captured |
| **Entity-centric QA** | Graph structure provides rich entity context |
| **Summarization of large corpora** | Community summaries enable corpus-level understanding |
| **Private enterprise data** | Build knowledge graph over internal documents |

### When NOT to Use GraphRAG

| Scenario | Why It's Overkill |
|:---------|:------------------|
| **Simple factual QA** | Standard RAG suffices for direct fact retrieval |
| **Single document QA** | No graph structure needed for one document |
| **Latency-critical apps** | Global search is too slow (400ms-2s+) |
| **Frequently changing data** | Re-indexing KG is expensive |
| **Budget constraints** | Indexing costs 10-50x more than standard RAG |
| **Low entity density** | If your corpus has few extractable entities, graph is sparse |

### Cost Analysis

GraphRAG has significantly higher costs than standard RAG:

| Phase | Cost Driver | Typical Scale |
|:------|:------------|:--------------|
| **Indexing** | LLM calls for entity extraction | ~1 call per chunk |
| **Indexing** | Gleaning (multi-pass extraction) | 2-3x extraction cost |
| **Indexing** | Community summarization | ~1 call per community |
| **Global Query** | Map phase (parallel) | 10-100 LLM calls |
| **Global Query** | Reduce phase | 1-3 LLM calls |
| **Local Query** | Subgraph retrieval + generation | 1-2 LLM calls |

**Rule of Thumb**: Indexing 1MB of text costs roughly $10-15 (depending on LLM pricing). Global queries consume 200K+ tokens.

### Indexing Cost Breakdown (Example)
For a corpus of 10,000 documents (~10MB of text):

```
Chunking: ~50,000 chunks (@ 600 tokens/chunk, 300 overlap)

Entity Extraction:
  - 50,000 LLM calls (1 per chunk)
  - With gleaning: +25,000 calls (50% additional)
  - ~75,000 calls total
  
Entity Resolution: CPU-only (minimal cost)

Community Detection: CPU-only (Leiden is efficient)

Community Summarization:
  - Depends on hierarchy depth
  - ~500-5,000 LLM calls for summaries

Embedding:
  - Standard embedding costs (~$0.0001/1K tokens)

Total: Heavy LLM dependency → $100-500+ for 10MB corpus
```

## Comparisons

### GraphRAG vs Standard RAG

| Aspect                  | Standard RAG                                        | GraphRAG                                     |
| :---------------------- | :-------------------------------------------------- | :------------------------------------------- |
| **Index Structure**     | Vector store (flat)                                 | Knowledge graph + communities (hierarchical) |
| **Retrieval Method**    | Cosine similarity                                   | Graph traversal + community lookup           |
| **Global Queries**      | Fails                                               | Map-reduce over summaries                    |
| **Multi-hop Reasoning** | Requires [[Multi-hop Reasoning\|explicit chaining]] | Implicit in graph structure                  |
| **Indexing Cost**       | Low (embeddings only)                               | High (LLM extraction + summarization)        |
| **Query Latency**       | ~100-200ms                                          | Local: ~200-400ms, Global: 1-5s              |
| **Data Freshness**      | Easy to update                                      | Expensive to re-index                        |
| **Explainability**      | Chunk citations                                     | Entity + relationship provenance             |

### GraphRAG vs Multi-hop Reasoning

| Aspect | [[Multi-hop Reasoning]] | GraphRAG |
|:-------|:------------------------|:---------|
| **Relationship Discovery** | Implicit (LLM infers) | Explicit (pre-extracted graph) |
| **Query-time Cost** | Multiple retrieval+LLM calls | Single subgraph retrieval |
| **Error Propagation** | Hop 1 error cascades | Relationships are fixed |
| **Setup Cost** | Low (standard RAG index) | High (knowledge graph construction) |
| **Best For** | Ad-hoc complex questions | Entity-relationship domains |

### Query Type Decision Matrix

| Query Type | Best Approach |
|:-----------|:--------------|
| "What is X?" (factual) | Standard RAG |
| "How are X and Y related?" | GraphRAG (local search) |
| "Who are all the people mentioned?" | GraphRAG (entity query) |
| "What are the main themes?" | GraphRAG (global search) |
| "What does X think about Y?" | Multi-hop or GraphRAG |
| "Summarize document Z" | Long context or standard RAG |

## Technical Deep Dive: Community Summarization

The community summarization step is critical for global search to work. At each level of the hierarchy:

```python
# Pseudo-code for community summary generation
def generate_community_summary(community):
    entities = get_entities_in_community(community)
    relationships = get_relationships_in_community(community)
    
    prompt = f"""
    You are analyzing a community of related entities.
    
    Entities:
    {format_entities(entities)}
    
    Relationships:
    {format_relationships(relationships)}
    
    Generate a comprehensive summary that captures:
    1. The main theme of this community
    2. Key entities and their roles
    3. Important relationships
    4. Any notable patterns or findings
    """
    
    return llm(prompt)
```

The resulting summaries serve as a "compressed reasoning context" that allows the LLM to reason about large portions of the corpus without processing individual chunks.

## Common Pitfalls

### 1. Entity Extraction Quality
**Problem**: Poor LLM prompts lead to missing or incorrect entities, corrupting the knowledge graph.

**Solution**:
- Use domain-specific few-shot examples in extraction prompts
- Implement gleaning (multi-pass extraction)
- Manual review of entity sample for quality assurance

### 2. Over-Granular Communities
**Problem**: Too many small communities make global search expensive and fragmented.

**Solution**:
- Tune Leiden resolution parameter
- Merge communities below a minimum size threshold
- Use appropriate hierarchy level for queries

### 3. Stale Knowledge Graphs
**Problem**: When source documents update, the knowledge graph becomes outdated.

**Solution**:
- Implement incremental indexing (update affected subgraphs only)
- Track document versions and their graph contributions
- Schedule periodic full re-indexing for data freshness

### 4. Global Search Latency
**Problem**: Map-reduce over many communities is slow (1-5+ seconds).

**Solution**:
- Pre-compute answers for common query patterns
- Use faster/cheaper LLMs for map phase
- Implement parallel processing for map calls
- Cache community-level partial answers

### 5. Cost Explosion at Scale
**Problem**: For very large corpora, indexing costs become prohibitive.

**Solution**:
- Use smaller/cheaper LLMs for extraction (with quality tradeoff)
- Sample documents for entity extraction, not 100% coverage
- Consider hybrid approaches: GraphRAG for core entities, standard RAG for supporting content

## Implementation Notes

### Key Libraries and Tools
- **Microsoft GraphRAG**: Official implementation (Python)
- **Neo4j**: Graph database for storing knowledge graphs
- **LangChain/LlamaIndex**: Orchestration frameworks with GraphRAG integrations
- **NetworkX / igraph**: Graph algorithms (Leiden community detection)

### Prompt Engineering for Entity Extraction

Example extraction prompt pattern:
```
Given the following text:
---
{text_chunk}
---

Extract all entities and relationships following this format:

ENTITIES:
- Name: [entity name]
  Type: [PERSON | ORGANIZATION | LOCATION | CONCEPT | EVENT]
  Description: [brief description based on the text]

RELATIONSHIPS:
- Source: [entity 1]
  Target: [entity 2]
  Relationship: [description of how they are related]

Be exhaustive. Include all entities mentioned, even if briefly.
```

## Variants and Extensions

### HybridRAG
Combines GraphRAG with standard vector retrieval:
- Use vector search for initial retrieval
- Expand results using graph relationships
- Best of both worlds: semantic similarity + structural relationships

### Incremental GraphRAG
Addresses the re-indexing problem:
- Track which documents contribute to which entities
- When a document updates, only re-extract affected subgraph
- Merge new entities with existing graph

### KET-RAG (Cost-Efficient GraphRAG)
Research from 2024 that reduces indexing costs:
- Multi-granular indexing (not everything needs LLM extraction)
- Selective entity extraction (high-information chunks only)
- Claims 10x cost reduction with comparable quality

## Resources

### Papers
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Microsoft, 2024)](https://arxiv.org/abs/2404.16130) - The original GraphRAG paper
- [Leiden Algorithm for Community Detection](https://www.nature.com/articles/s41598-019-41695-z) - The community detection method used

### Articles
- [Microsoft Research Blog: GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [DataCamp: GraphRAG Tutorial](https://www.datacamp.com/tutorial/graphrag)

### Code
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag) - Official implementation
- [Neo4j GraphRAG Examples](https://neo4j.com/developer-blog/graphrag-knowledge-graph-rag/)

### Videos
- [GraphRAG Explained (Weights & Biases)](https://www.youtube.com/results?search_query=graphrag+weights+and+biases)

---

**Back to**: [[01 - RAG Index]]
