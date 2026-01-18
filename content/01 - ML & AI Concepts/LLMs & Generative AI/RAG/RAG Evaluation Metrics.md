## Overview

Evaluating RAG systems is fundamentally different from evaluating traditional ML models. Unlike classification or regression where you have ground truth labels, RAG evaluation must assess a complex pipeline with multiple failure modes: retrieval quality, context utilization, answer generation, and faithfulness to source material.

**How do you measure if an LLM correctly used the retrieved context to answer a question, and whether the retrieved context was even the right information to retrieve?**

RAG evaluation has evolved from simple retrieval metrics (Precision@K, Recall@K) to sophisticated LLM-as-a-judge frameworks that assess nuanced dimensions like faithfulness, context relevance, and answer quality. Modern evaluation frameworks like [[RAGAS]], TruLens, and Phoenix (Arize) provide automated pipelines for these assessments.



## The RAG Triad

![[RAG Evaluation Metrics 2026-01-10 15.20.02.excalidraw.svg]]

1. **Context Relevance**: Did retrieval find the right information?
2. **Faithfulness/Groundedness**: Does the answer stick to what was retrieved?
3. **Answer Relevance**: Did the answer actually address the question?

A failure in any of above breaks the system:
- Poor Context Relevance → Garbage in, garbage out
- Poor Faithfulness → Hallucinations despite good context
- Poor Answer Relevance → Correct information but wrong focus

## Why Traditional Metrics Fall Short

Traditional retrieval metrics like `Precision@K` and `Recall@K` require ground truth labels for every query-document pair. In production RAG systems:
- You rarely have labeled relevance judgments
- "Relevance" is nuanced (a document can be partially relevant)
- The final answer quality matters more than intermediate retrieval metrics

Hence modern RAG evaluation uses **LLM-as-a-judge**: using language models to assess quality along multiple dimensions, with or without ground truth.

## Component-wise *and* E2E Evaluation

Two complementary evaluation philosophies.

|                    | What It Measures                          | Advantage                | Limitation                   |
| :----------------- | :---------------------------------------- | :----------------------- | :--------------------------- |
| **Component-wise** | Individual stages (retrieval, generation) | Pinpoints failure modes  | Doesn't capture interactions |
| **End-to-End**     | Final answer quality                      | Measures real user value | Harder to debug failures     |

Best practice: Use both. Component metrics diagnose problems; end-to-end metrics validate fixes.


## Core Metrics

### 1. Context Relevance (Retrieval Quality)

"How relevant is the retrieved context to the user's query?"
This metric catches retrieval failures before they propagate to generation.

#### A. LLM-as-Judge
An LLM extracts sentences from the context that are relevant to the query, then computes the ratio:

$$\text{Context Relevance} = \frac{\text{Number of relevant sentences in context}}{\text{Total sentences in context}}$$

**Prompt Template** (simplified):
```
Given the following question and context, extract sentences from the context
that are directly relevant to answering the question.

Question: {question}
Context: {context}

Relevant sentences (return only exact quotes, or "None" if no relevant sentences):
```


#### B. Traditional Retrieval Metrics
When ground truth exists

**Precision@K**: What fraction of the top-K retrieved documents are relevant?

$$\text{Precision@K} = \frac{|\{\text{relevant docs}\} \cap \{\text{retrieved docs}\}|}{K}$$

**Recall@K**: What fraction of all relevant documents did we retrieve?

$$\text{Recall@K} = \frac{|\{\text{relevant docs}\} \cap \{\text{retrieved docs}\}|}{|\{\text{all relevant docs}\}|}$$

**Mean Reciprocal Rank (MRR)**: Where does the first relevant document appear?

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position of the first relevant document for query $i$.


- Use [[#A. LLM-as-Judge)|LLM-as-Judge]] when you lack ground truth labels
- Use [[#B. Traditional Retrieval Metrics|Precision, Recall, MRR]] when you have labeled query-document pairs


### 2. Faithfulness (Groundedness)

Can every claim in the generated answer be traced back to the retrieved context?

The LLM is supposed to be a "faithful summarizer" of the context, not a creative writer. If the answer says "The product launched in 2019" but the context says "The product launched in 2020," that is a faithfulness violation (hallucination).

This metric catches hallucinations. Even if the answer sounds plausible and addresses the question, if it contains information not in the context, the RAG system has failed its core promise.

#### Computation (RAGAS Approach)

1. **Extract claims**: Break the answer into atomic statements/claims
2. **Verify each claim**: Check if each claim is supported by the context
3. **Compute ratio**:

$$\text{Faithfulness} = \frac{\text{Number of claims supported by context}}{\text{Total claims in answer}}$$

**Example Claim Extraction Template**:
```
Break down the following answer into independent, atomic statements
(claims that can be verified individually).

Answer: {answer}

Atomic statements:
1. ...
2. ...
```

**Example Claim Verification Template**:
```
Given the following context and statement, determine if the statement
can be inferred from the context.

Context: {context}
Statement: {statement}

Verdict (Yes/No):
Reasoning:
```

#### Variations

**Binary Faithfulness**: Is the answer fully faithful? (Yes/No)
- Stricter, can be used when _any_ hallucination is unacceptable (legal, medical)

**Weighted Faithfulness**: Weight claims by importance (TODO: How?)
- Critical claims (dates, numbers, names) weighted higher than stylistic claims

### 3. Answer Relevance

**Definition**: Does the generated answer actually address the user's question?

**Intuition**: You asked "What is the capital of France?" and the system responds with accurate, well-grounded information about French cuisine. The context might be relevant, the answer might be faithful, but it does not answer what was asked.

#### Computation Approach

Generate hypothetical questions that the answer would be a good response to, then measure similarity to the original question:

1. **Generate reverse questions**: Given the answer, what questions would it answer?
2. **Compare to original**: How similar are the generated questions to the original query?

$$\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{e}_q, \mathbf{e}_{q_i})$$

Where $\mathbf{e}_q$ is the original query embedding and $\mathbf{e}_{q_i}$ are embeddings of generated questions.

#### Direct LLM Scoring

```
On a scale of 1-5, how well does the following answer address the question?

Question: {question}
Answer: {answer}

Score:
Reasoning:
```


### 4. Answer Correctness (When Ground Truth Available)

How factually correct is the answer compared to a known ground truth?

If we have a labeled test set with expected answers, this measures how close the generated answer is to the expected answer.


Combines semantic similarity with factual overlap:

$$\text{Answer Correctness} = w_1 \cdot F_1 + w_2 \cdot \text{Semantic Similarity}$$

Where:
- $F_1$ measures word/phrase overlap between generated and ground truth answers
- Semantic Similarity uses embedding cosine similarity

**Token-level F1 Score**:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where:
- Precision = fraction of generated tokens that appear in ground truth
- Recall = fraction of ground truth tokens that appear in generated answer


### 5. Context Recall (When Ground Truth Available)

Given a ground truth answer, can we attribute each statement in the ground truth to the retrieved context?

This is a retrieval quality metric that requires ground truth. It checks: "If we knew the correct answer, did our retrieval actually find the information needed to produce that answer?"

$$\text{Context Recall} = \frac{\text{GT statements attributable to context}}{\text{Total GT statements}}$$

**Use case**: Diagnosing retrieval failures. If Context Recall is low, retrieval is missing critical information, and no amount of prompt engineering will fix it.


### 6. Context Precision

Are the relevant chunks ranked higher than irrelevant ones in the retrieved results?

Even if you retrieve all relevant documents, if they are ranked 8th, 9th, and 10th while irrelevant documents are ranked 1st, 2nd, and 3rd, the LLM may focus on the wrong information.

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total relevant items in top-K}}$$

Where $v_k = 1$ if the item at rank $k$ is relevant, else 0.

This metric penalizes relevant documents appearing late in the ranking.


## Other Evaluation Concepts

### Noise Robustness

How well does the RAG system perform when irrelevant context is injected?

In production, retrieval often returns some irrelevant documents. A robust RAG system should ignore noise and focus on relevant context.

1. Take a query with known relevant context
2. Inject N irrelevant documents into the context
3. Measure faithfulness and answer relevance
4. Compare to baseline (no noise injection)


$$\text{Noise Robustness} = \frac{\text{Performance with noise}}{\text{Performance without noise}}$$


### Negative Rejection

Does the system correctly abstain from answering when the context does not contain the answer?

A good RAG system should say "I don't know" when the retrieved context genuinely lacks the information, rather than hallucinating an answer.

1. Create test cases where the context deliberately lacks the answer
2. Check if the system responds with uncertainty or an abstention
3. Measure the false answer rate

$$\text{Negative Rejection Rate} = \frac{\text{Correct abstentions}}{\text{Total unanswerable questions}}$$

### Counterfactual Robustness

Does the system correctly use the retrieved context even when it contradicts the LLM's parametric knowledge?

If the context says "The Eiffel Tower is in London" (counterfactual), a faithful RAG system should report this (it is being faithful to context), or flag the contradiction, rather than defaulting to its training data.

1. Create test cases with counterfactual contexts
2. Measure if the answer reflects the context (faithful) or the LLM's prior knowledge (unfaithful)

This tests whether the RAG system truly grounds on context vs. using retrieval as a mere prompt trigger.


### Information Integration

Can the system synthesize information across multiple retrieved chunks?

1. Create questions requiring multi-source synthesis
2. Measure if the answer correctly integrates all sources
3. Check for omissions or incorrect combinations


## LLM-as-a-Judge: Deep Dive

### Why Use LLMs for Evaluation?

Traditional metrics require ground truth labels, which are:
- Expensive to create (human annotation)
- Not always available (novel questions)
- Subjective (what counts as "relevant"?)

LLMs can approximate human judgment at scale, with several advantages:
- No labeled data required (reference-free evaluation)
- Can assess nuanced qualities (coherence, helpfulness)
- Scales well

### Challenges and Mitigations

| Challenge                | Description                            | Mitigation                                         |
| :----------------------- | :------------------------------------- | :------------------------------------------------- |
| **Self-preference bias** | LLMs prefer their own outputs          | Use different models for generation vs. evaluation |
| **Position bias**        | Order of options affects judgment      | Randomize option order, average across orderings   |
| **Verbosity bias**       | Longer answers rated higher            | Normalize by length or instruct to ignore length   |
| **Inconsistency**        | Same input, different outputs          | Use low temperature, average multiple runs         |
| **Sycophancy**           | Agrees with user's implicit preference | Use neutral prompts, avoid leading questions       |

1. **Use structured outputs**: Force JSON or specific formats to parse reliably
2. **Chain-of-thought**: Ask for reasoning before the verdict
3. **Calibration**: Test on known examples to validate the judge
4. **Multiple judges**: Average scores from multiple LLM calls
5. **Human validation**: Periodically validate LLM judgments against human labels

## Evaluation Frameworks

### RAGAS (Retrieval Augmented Generation Assessment)

Most widely adopted open-source RAG evaluation framework. Provides reference-free metrics using LLM-as-a-judge.

**Core Metrics Provided**:
- Context Relevancy
- Context Recall (requires ground truth)
- Faithfulness
- Answer Relevancy
- Answer Semantic Similarity (requires ground truth)
- Answer Correctness (requires ground truth)

**Key Features**:
- Reference-free evaluation
- Synthetic test set generation
- Supports multiple LLM providers

**Limitations**:
- Depends on LLM quality (garbage in, garbage out)
- Can be slow for large datasets (many LLM calls)
- Metric definitions may not match your specific use case

### TruLens

**Overview**: Evaluation and observability platform for LLM applications, with strong support for RAG.

**Key Features**:
- **Feedback Functions**: Modular metrics you can customize
- **Tracing**: Full pipeline observability (retrieval latency, LLM calls)
- **Dashboards**: Visual exploration of evaluation results
- **Guardrails**: Real-time monitoring in production

**Core Metrics**:
- Groundedness (faithfulness)
- Context Relevance
- Answer Relevance
- Toxicity, Bias, and other safety metrics

**Strengths**:
- Production-ready with monitoring
- Customizable feedback functions
- Good LlamaIndex integration

### Phoenix (Arize AI)

Open-source observability platform with strong evaluation capabilities.

**Key Features**:
- **Tracing**: Visualize the entire RAG pipeline
- **Embeddings Analysis**: Cluster and explore retrieval quality
- **LLM Evals**: Built-in evaluation using LLM-as-judge
- **Drift Detection**: Monitor embedding drift over time

**Core Evaluation Capabilities**:
- Relevance classification (binary or graded)
- Hallucination detection
- Q&A correctness
- Custom evaluation templates

**Strengths**:
- Excellent visualization
- Strong embedding analysis
- Works well for debugging retrieval

### Comparison of Frameworks

| Feature | RAGAS | TruLens | Phoenix |
|:---|:---|:---|:---|
| **Focus** | Evaluation metrics | Eval + Observability | Observability + Eval |
| **Reference-free** | Yes | Yes | Yes |
| **Synthetic data gen** | Yes | Limited | No |
| **Production monitoring** | Limited | Yes | Yes |
| **Visualization** | Basic | Dashboard | Excellent |
| **Customization** | Moderate | High | High |
| **Learning curve** | Low | Medium | Medium |
| **Best for** | Quick evaluation | Production apps | Debugging retrieval |

## Synthetic Test Set Generation

### The Evaluation Data Problem

Proper RAG evaluation requires test data with:
- Questions
- Ground truth answers
- Relevant documents (sometimes)

Creating this manually is expensive. LLMs can be used to generate synthetic test sets from your corpus.

### How It Works
![[RAG Evaluation Metrics 2026-01-10 16.46.19.excalidraw.svg]]

### Generation Strategies

#### 1. Simple Question Generation
Generate straightforward questions from chunks:
```
Given the following text, generate 3 questions that can be answered
using only the information in the text.

Text: {chunk}

Questions:
```

#### 2. Multi-hop Question Generation
Generate questions requiring multiple chunks:
```
Given the following two text passages, generate a question that
requires information from BOTH passages to answer.

Passage 1: {chunk1}
Passage 2: {chunk2}

Question:
Answer:
```

#### 3. Reasoning Question Generation
Generate questions requiring inference:
```
Given the following text, generate a question that requires
reasoning or inference beyond simple fact retrieval.

Text: {chunk}

Question:
Expected reasoning steps:
Answer:
```

### Quality Control for Synthetic Data

**Problem**: LLM-generated questions can be:
- Too easy (answer is a direct quote)
- Too hard (requires external knowledge)
- Ambiguous or poorly phrased
- Factually incorrect

**Mitigations**:
1. **Filtering**: Use another LLM to filter low-quality questions
2. **Human spot-checks**: Validate a random sample
3. **Diversity constraints**: Ensure question type variety
4. **Answer verification**: Check that the answer is actually in the source

**Quality Filter Prompt**:
```
Evaluate the following question-answer pair for quality.

Question: {question}
Answer: {answer}
Source text: {chunk}

Rate on 1-5 scale for:
1. Clarity: Is the question clear and unambiguous?
2. Answerability: Can the question be answered from the source?
3. Difficulty: Is it neither too trivial nor requiring external knowledge?

Overall quality score (1-5):
Should include in test set (Yes/No):
```


## Practical Application

### When to Use Which Metrics

|                              | Primary Metrics                         | Secondary Metrics     |
| :--------------------------- | :-------------------------------------- | :-------------------- |
| **Quick health check**       | Faithfulness, Answer Relevance          | Context Relevance     |
| **Retrieval debugging**      | Context Relevance, Context Precision    | Recall@K              |
| **Hallucination monitoring** | Faithfulness                            | Negative Rejection    |
| **Production monitoring**    | Faithfulness, Answer Relevance, Latency | Error rates           |
| **A/B testing**              | Answer Correctness, User satisfaction   | All component metrics |
| **Chunking optimization**    | Context Relevance, Context Recall       | Faithfulness          |

### Evaluation Pipeline Design

![[RAG Evaluation Metrics 2026-01-10 16.51.57.excalidraw.svg]]


### Common Pitfalls

#### 1. Evaluating Only End-to-End
If the final answer is wrong, you do not know if retrieval or generation failed.
Always include component metrics alongside end-to-end metrics.

#### 2. Ignoring Edge Cases
Average metrics look good, but the system fails catastrophically on certain query types.

Potential Solutions:
- Segment evaluation by query type (simple, multi-hop, keyword-heavy)
- Analyze the tail (worst 10% of results)
- Include negative test cases (unanswerable questions)

#### 3. Overfitting to Synthetic Data
System performs well on LLM-generated questions but fails on real user queries.

- Include real user queries in test set (with privacy considerations)
- Validate synthetic questions for realism
- Periodically refresh test sets

#### 4. Inconsistent Evaluation Conditions
Comparing systems evaluated under different conditions (different LLM judges, temperatures, etc.)

- Document evaluation setup
- Use deterministic settings (low temperature)
- Version control evaluation configurations



## Relationship Between Metrics

### Metric Correlations and Trade-offs

```
High Context Relevance + Low Faithfulness
= LLM ignoring good context (prompt issue or model issue)

Low Context Relevance + High Faithfulness
= LLM faithfully reporting irrelevant info (retrieval issue)

High Faithfulness + Low Answer Relevance
= Correct info, wrong focus (question understanding issue)

Low Context Relevance + Low Faithfulness + Low Answer Relevance
= Everything is broken (start with retrieval)
```

### Diagnostic Flowchart

![[RAG Evaluation Metrics 2026-01-10 17.07.49.excalidraw.svg]]


## Resources

### Papers
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476)
- [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431)

### Tools and Libraries
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [TruLens](https://github.com/truera/trulens)
- [Phoenix (Arize)](https://github.com/Arize-ai/phoenix)
- [LangSmith](https://www.langchain.com/langsmith) 
- [DeepEval](https://github.com/confident-ai/deepeval)

### Others
- [Evaluating RAG Applications with RAGAs (LangChain Blog)](https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/)
- [Pinecone: RAG Evaluation](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)


---

**Back to**: [[01 - RAG Index]]
