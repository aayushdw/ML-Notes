## Overview
**Multi-hop Reasoning** is an advanced RAG technique that chains multiple retrieval and reasoning steps to answer questions that cannot be solved with a single retrieval pass. Instead of retrieving once and generating an answer, the system iteratively retrieves information, reasons about intermediate results, and performs additional retrievals based on that reasoning.

It solves the problem of **"one retrieval isn't enough"** by introducing a feedback loop: retrieve → reason → retrieve again → reason → generate.

## Core Concept

### The Problem It Solves
Standard RAG retrieves relevant chunks once and generates an answer:
![[Multi-hop Reasoning 2026-01-13 18.04.11.excalidraw.svg]]

Multi-hop reasoning recognizes that some questions require **sequential knowledge dependencies**:

![[Multi-hop Reasoning 2026-01-13 18.07.50.excalidraw.svg]]

Multi-hop reasoning breaks down complex questions into decomposable sub-questions, where answering one question provides context for the next. This is closest to how humans solve complex problems.

It is a sequence of (Retrieval, Reasoning) pairs executed iteratively:
1. **First Hop**: Retrieve documents relevant to the original question
2. **Reason**: Use LLM to identify what additional information is needed
3. **Second Hop**: Retrieve documents based on intermediate reasoning
4. **Repeat**: Continue until sufficient context is gathered
5. **Final Generation**: Synthesize all gathered context into final answer


## Execution Patterns

### Pattern 1: Sequential Decomposition (Explicit)
The LLM explicitly breaks down the question into sub-questions:

```
Step 1: Parse Question
  Input: "What is the climate impact of the company that invented solar panels?"

Step 2: LLM Decomposes
  Sub-question 1: "Who invented solar panels?"
  Sub-question 2: "What is the climate impact of [company from answer 1]?"

Step 3: Execute Hops
  Hop 1: Retrieve docs for "solar panel inventor"
         - Result: "SunPower Corporation"

  Hop 2: Retrieve docs for "SunPower climate impact"
         - Result: "Reduced 500M tons CO2 annually"

Step 4: Generate Answer
  "SunPower Corporation, which pioneered modern solar panels, has reduced
   carbon emissions by 500M tons annually..."
```

### Pattern 2: Implicit Multi-hop (Agentic)
The system uses an agent loop to decide when to retrieve again:

```
Iteration 1:
  ├─ Retrieve once based on original query
  ├─ LLM generates partial answer
  └─ LLM decides: "I need more info about [X]"

Iteration 2:
  ├─ Retrieve again based on identified gap
  ├─ LLM refines answer
  └─ LLM decides: "I have enough info"

Output: Final synthesized answer
```

> [!NOTE] Relationship to Agentic RAG
> This pattern represents the **iterative retrieval loop** component of [[Agentic RAG]]. This pattern focuses specifically on the "retrieve → reason → retrieve again" cycle, which is a core building block of agentic architectures.

### Pattern 3: Graph-based Traversal
Uses entity relationships to navigate knowledge:

```
Query: "What funding did the CTO of OpenAI's founded company raise?"

Graph Traversal:
  OpenAI → (CTO: Sam Altman) → Sam Altman's companies → Funding info
           → (Founded: Sam Altman) → Y Combinator → Funding amounts
```

> [!NOTE] Relationship to GraphRAG
> This pattern describes graph traversal *at query time*, assuming a knowledge graph already exists. This is essentially what [[GraphRAG]]'s **Local Search** does. However, GraphRAG is a complete system that also handles graph *construction* (entity extraction, community detection, hierarchical summarization) and supports additional query modes like Global Search for corpus-wide questions.


## Architecture Patterns

### 1. **Explicit Decomposition Pipeline**
Best for: Well-structured questions with clear sub-goals

```
Query → LLM Decomposer → [Sub-Q1, Sub-Q2, Sub-Q3]
        ↓
       Parallel Retrieval for each Sub-Q
        ↓
       Context Aggregator
        ↓
       Final LLM Generator → Answer
```

**Pros**: Predictable, easy to debug, parallelizable
**Cons**: Requires good decomposition prompt, fails on ambiguous questions
**Cost**: ~N retrieval calls (N = # sub-questions)

### 2. **Agentic/Iterative Loop**
Best for: Open-ended questions, exploratory reasoning

```
Query → LLM Agent with Retrieval Tool
  │
  ├─ Tool Call: retrieve("query refinement 1")
  ├─ Observe: [context 1]
  ├─ Reason: "Need more about X"
  │
  ├─ Tool Call: retrieve("focused query 2")
  ├─ Observe: [context 2]
  ├─ Reason: "Sufficient info"
  │
  └─ Final Response
```

**Pros**: Adaptive, handles unexpected paths, good for complex reasoning
**Cons**: Variable latency, cost unpredictable, harder to debug

### 3. **Hierarchical/Tree Search**
Best for: Questions with branching dependencies

```
                    Original Query
                          │
              ┌───────────┼───────────┐
              │           │           │
            Sub-Q1      Sub-Q2      Sub-Q3
              │           │           │
            ┌─┴─┐       │         ┌─┘
          Sub-Q1a  Sub-Q1b        │
```

**Pros**: Handles complex dependencies, can prune irrelevant branches
**Cons**: Expensive (exponential retrieval), complex orchestration
**Cost**: ~O(branching_factor^depth) retrieval calls

## When to Use Multi-hop Reasoning

### Use Multi-hop When:

**Questions have implicit dependencies**
- "Who funded the company that created GPT?" (Company → Founder → Founder's investors)
- "What regulations apply to this industry's main competitor?" (Industry → Competitors → Competitor regulations)
- "How does the technology from [X] relate to [Y]?" (X details → X connections → Y details)

**Retrieval shows gaps**
- Single retrieval returns "Company: TechCorp" but user needs "Company: TechCorp, Founded by: Jane Doe"
- Retrieved context references entities not yet explained

**Questions require comparison or synthesis**
- "Compare the founding philosophies of companies A and B" (Retrieve A → Retrieve B → Compare)
- "How do these three methodologies relate?" (Retrieve 1 → Retrieve 2 → Retrieve 3 → Synthesis)

**Domain knowledge graph is sparse**
- Without explicit relationships, multi-hop traversal discovers them implicitly
- E.g., medical: symptoms → conditions → treatments → side effects

### Don't Use Multi-hop When:

**Questions are factual, single-retrieval answerable**
- "What is the capital of France?" (Paris - one retrieval sufficient)
- "Who invented the telephone?" (Alexander Graham Bell - direct fact)
- Cost/latency overhead not justified

**Latency requirements are strict**
- Multi-hop adds retrieval latency linearly (or exponentially in tree search)
- If p99 latency < 500ms, multi-hop is risky (each retrieval ~100-300ms)

**Vector database/retrieval quality is poor**
- Garbage In, Garbage Out: Bad retrieval at hop 1 cascades to worse retrieval at hop 2
- Fix retrieval quality first before attempting multi-hop

## Production Considerations

### Latency & Cost Trade-offs

| Aspect                         | Single Retrieval | Multi-hop (2 hops)          | Multi-hop (3+ hops) |
| :----------------------------- | :--------------- | :-------------------------- | :------------------ |
| **Retrieval Calls**            | 1                | 2-3                         | 3-5+                |
| **Typical Latency**            | 150ms            | 300-450ms                   | 450-700ms+          |
| **Vector DB Cost**             | 1 call           | 2-3 calls                   | 3-5+ calls          |
| **LLM Cost**                   | 1 generation     | 2-3 generations (reasoning) | 3-5+ generations    |
| **Answer Quality (potential)** | Good             | Better                      | Best                |

**Production Rule**: Multi-hop could add ~150-200ms per additional hop. Budget accordingly.

### Implementation Challenges

#### 1. **Context Explosion**
With each retrieval, accumulated context grows. By hop 3-4, you might exceed LLM context windows.

*Solution*:
- Use context compression (summarize previous hops)
- Implement context window budgeting (reserve 30% for final generation)
- Track context relevance and prune irrelevant chunks before next hop

```python
# Example: Budget-aware context management
max_context_tokens = 4000
reserved_for_generation = max_context_tokens * 0.3
available_for_retrieval = max_context_tokens * 0.7

for hop in range(max_hops):
    remaining_budget = available_for_retrieval - sum(tokens_per_chunk)
    if remaining_budget < 200:  # Minimum viable chunk size
        break
    retrieve_next(budget=remaining_budget)
```

#### 2. **Information Consistency**
Different hops might retrieve contradictory information, especially if docs are stale.

*Solution*:
- Track document timestamps and flag outdated sources
- Use conflict detection: "Documents X and Y contradict. Which is more recent?"
- Implement consensus mechanisms (prefer agreement across multiple sources)

#### 3. **Determining Hop Count**
How many hops are enough? Too few → incomplete answers. Too many → cost/latency explosion.

*Solution*:
- **Fixed**: Set max hops based on domain (e.g., "financial reasoning needs max 3 hops")
- **Adaptive**: Use stopping criteria:
  - LLM signals "I have enough info"
  - Context relevance plateaus (next retrieval adds <5% novel info)
  - Token budget exhausted
  - Confidence threshold reached

#### 4. **Query Degradation**
As you compose queries for subsequent hops, they might drift from original intent or become too specific/vague.

*Solution*:
- Keep original query in context (reference: "Given the original question about X...")
- Use query refinement: Generate next query using LLM but validate it's related
- Test query similarity to original (if cosine similarity < 0.3, flag as drift)

### Monitoring & Observability

For production multi-hop systems, track:

1. **Hop Count Distribution**
   - What % of queries need 1 hop? 2? 3+?
   - If most need 3+, your chunking strategy may be poor

2. **Context Relevance per Hop**
   - Calculate: How relevant is each retrieved chunk to the original query?
   - If relevance drops significantly at hop 2+, you're accumulating noise

3. **Latency Breakdown**
   - Log: Time per retrieval, Time per reasoning, Total end-to-end
   - Identify bottleneck (retrieval vs LLM inference)

4. **Answer Quality Metrics**
   - Compare: Single-hop answer vs multi-hop answer for same question
   - Measure: Improvement in correctness, comprehensiveness, user satisfaction

### Example Monitoring Dashboard
```
Multi-hop Reasoning Metrics:
├─ Avg Hops per Query: 2.1
├─ 1-Hop Queries: 35%
├─ 2-Hop Queries: 45%
├─ 3+-Hop Queries: 20%
├─ P50 Latency: 320ms
├─ P99 Latency: 680ms
├─ Retrieval Quality (Context Relevance): 0.78
├─ Answer Faithfulness: 0.92
└─ User Satisfaction (CSAT): 4.2/5.0
```

## Practical Implementation Techniques

### 1. **Self-Ask Pattern** (Simple Explicit Decomposition)
Used by systems like WebGPT:

```
Q: "What is the capital of the country that invented the telephone?"

Model output:
"First, I need to find: Who invented the telephone?
 Searching for: 'who invented telephone'
 Result: Alexander Graham Bell from Scotland.

 Now I need to find: What is Scotland's capital?
 Searching for: 'capital of Scotland'
 Result: Edinburgh.

 Answer: Edinburgh is the capital of Scotland, where the telephone was invented."
```

### 2. **Re-Act (Reasoning + Acting)**
Combines explicit reasoning with tool use:

```
Thought: I need to find who built GPT, then find their funding sources.
Action: retrieve("who created GPT")
Observation: [contexts about OpenAI, Sam Altman, etc.]

Thought: Now I need to find OpenAI's funding.
Action: retrieve("OpenAI funding sources investors")
Observation: [contexts about funding rounds]

Thought: I have enough information.
Final Answer: [Synthesized response]
```

### 3. **Graph-based Iteration** (Entity-Aware)
Track retrieved entities and follow connections:

```
Query: "What is the CEO's educational background at TechCorp?"

Retrieved Entities:
├─ TechCorp
├─ TechCorp.CEO = Jane Doe
└─ Jane Doe.Education = MIT, Computer Science

Next Hop Triggers:
- If needed: Retrieve more about Jane Doe's achievements
- If needed: Retrieve about MIT CS program
```

## Comparison with Alternatives

| Approach | Complexity | Latency | Answer Quality | Cost | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Retrieval RAG** | Low | Low (~150ms) | Good | Low | Factual Q&A, High SLA |
| **Multi-hop Explicit** | Medium | Medium (~350ms) | Better | Medium | Structured domains, Known deps |
| **Multi-hop Agentic** | High | Variable (~400-700ms) | Best | High | Complex reasoning, Exploration |
| **Fine-tuned LLM** | Very High | Low | Very Good | Very High | Domain-specific, High freq |
| **Long Context (100K tokens)** | Medium | Medium | Good | High | Document-heavy, Single source |

## Common Pitfalls

### 1. **Unlimited Hops**
Without stopping criteria, systems fetch 5-10 hops unnecessarily.
- **Fix**: Always set max_hops = 3 in production (diminishing returns after)

### 2. **Query Drift**
Each hop's query becomes increasingly specific, losing the original intent.
- **Fix**: Always include "relative to the original question about [X]" in prompts

### 3. **Context Overload**
By hop 3, accumulated context exceeds token limits, truncating valuable info.
- **Fix**: Use context ranking/compression. Keep only top 3 chunks per hop.

### 4. **Slow Cascading Failures**
Retrieval at hop 1 returns garbage → hop 2 searches for irrelevant terms → bad final answer.
- **Fix**: Validate retrieval quality per hop. Fallback to single-hop if relevance < threshold.

### 5. **Hallucination Compounding**
LLM hallucinates an entity at hop 1 → hop 2 retrieves noise related to hallucination.
- **Fix**: Use grounding checks ("Is [entity] mentioned in any document?")

## Real-world Examples

### Example 1: Customer Support (E-commerce)
```
Customer: "I bought a Samsung TV from Store X but the warranty was voided.
           What's the policy on manufacturer warranties if third parties void them?"

Hop 1: Retrieve about Store X's warranty policy
       → "Store X covers manufacturer defects for 2 years"

Hop 2: Retrieve about Samsung's warranty terms and third-party voiding
       → "Samsung voids warranty if non-Samsung parts installed"

Hop 3: Retrieve about local consumer protection laws
       → "Local law: Non-manufacturer actions can't void consumer protections"

Answer: Synthesize: "While Samsung's warranty is voided by third parties,
         local consumer protection laws may still require Store X to honor
         coverage for original defects..."
```

### Example 2: Medical/Legal Research
```
Doctor: "Are there case studies of Drug X interactions with Condition Y
         specifically in patients over 65?"

Hop 1: Retrieve about Drug X side effects and interactions
       → "Drug X contraindicated with medications for Condition Y"

Hop 2: Retrieve about specific research in elderly patients (65+)
       → "Study published 2023: Drug X shows 40% adverse event rate in 65+"

Hop 3: Retrieve case studies from that research
       → [Specific patient cases and outcomes]

Answer: "Yes, recent studies show Drug X has significant interactions
         with Condition Y treatments in patients 65+, with documented cases..."
```

## When to Consider Alternatives

### Use **Single-Hop RAG** if:
- Questions are primarily factual
- 95% of answers answerable with single retrieval
- Strict latency requirements (< 300ms)
- Cost is primary constraint

### Use **Fine-tuning** if:
- Your domain has consistent patterns
- You have high volume of similar questions
- Latency is critical
- You want to avoid external knowledge dependencies

### Use **GraphRAG** if:
- Your knowledge is highly relational
- Entities and their connections matter
- You have structured data available
- Complex entity-centric queries are common

### Use **Long Context Windows** if:
- Questions relate to single documents
- You can retrieve entire documents
- Context coherence is critical
- Latency allows (larger prompts = slower inference)

## Production Deployment Checklist

- [ ] **Latency budgeting**: Set max latency per hop, total latency < SLA
- [ ] **Cost analysis**: Calculate retrieval cost per hop, acceptable rate
- [ ] **Stopping criteria**: Define when to stop retrieving (confidence, budget, max hops)
- [ ] **Context management**: Implement compression/ranking to handle context explosion
- [ ] **Quality monitoring**: Track context relevance, answer correctness per hop count
- [ ] **Fallback strategy**: If multi-hop latency exceeds threshold, fall back to single-hop
- [ ] **Error handling**: What happens if retrieval fails at hop 2? Graceful degradation?
- [ ] **Prompt tuning**: Test decomposition prompts, stopping prompts, synthesis prompts
- [ ] **Query validation**: Detect and prevent query drift, hallucination propagation
- [ ] **User feedback loop**: Collect user ratings to assess if multi-hop is helping

## Resources & Further Reading

- **Paper**: [Self-Ask with Language Models](https://arxiv.org/abs/2210.03350) - Explicit decomposition approach
- **Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Reasoning + tool use
- **Paper**: [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625) - Decomposition strategies
- **Blog**: [LangChain Agent Loops](https://python.langchain.com/docs/modules/agents/) - Practical implementation
- **Related**: [[GraphRAG]] - When multi-hop relationships are explicit in a knowledge graph
- **Related**: [[RAG (Retrieval Augmented Generation) Overview]] - Parent concept

## Personal Notes
*[Space for your thoughts and learnings...]*

## Progress Checklist
- [ ] Understand single-hop limitations
- [ ] Grasp multi-hop decomposition patterns
- [ ] Learn production trade-offs (latency, cost, quality)
- [ ] Review implementation patterns (Explicit, Agentic, Graph-based)
- [ ] Study production challenges (context explosion, consistency, drift)
- [ ] Hands-on practice (Build explicit decomposition RAG)
- [ ] Evaluate when multi-hop is worth the cost

**Back to**: [[01 - ML & AI Concepts/LLMs & Generative AI/RAG (Retrieval Augmented Generation) Overview]]
