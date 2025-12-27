## Overview
Chunking is the process of breaking down large documents into smaller, manageable pieces (chunks) that fit within an LLM's context window.

It adheres to the **Goldilocks Principle**:
*   **Too Small**: You lose context. (e.g., "He said yes." -> Who is he? Yes to what?)
*   **Too Big**: You introduce noise, dilute specific information, and hit token limits.

## Core Strategies

### 1. Fixed-Size Chunking
The most basic approach. Splits text after a specific number of characters or tokens.
*   **Mechanism**: `if len(chunk) > limit: split()`
*   **Overlap**: Essential to prevent cutting words in half at the boundary.
*   **Pros**: computationally cheap, simple to implement.
*   **Cons**: Breaks semantic meaning mid-sentence.
*   **Production Readiness**: **Low** (Avoid unless resources are extremely constrained).

### 2. Recursive Character Chunking
The distinct standard for most text-based RAG applications. It tries to split on natural boundaries first.
*   **Mechanism**: It looks for separators in a specific order: `["\n\n", "\n", " ", ""]`.
    1.  Can I split by paragraphs (`\n\n`)?
    2.  If still too big, try lines (`\n`).
    3.  If still too big, try spaces.
*   **Pros**: Respects document structure and keeps paragraphs together.
*   **Cons**: Might still break semantic flow if a topic spans multiple paragraphs.
*   **Production Readiness**: **High** (The default choice).

### 3. Document Specific Chunking
Leverages the structure of known file types.
*   **Markdown**: Splits by headers (`#`, `##`). Ensures a Header and its contents stay together.
*   **Code**: Splits by class and function definitions (`class`, `def`).
*   **Pros**: Extremely high context preservation.
*   **Production Readiness**: **High** (Must-have for specific domains).

### 4. Semantic Chunking
Uses the *meaning* of the text to decide where to split, rather than arbitrary characters. It attempts to keep topically related sentences in the same chunk.

#### The Algorithm
1.  **Sentence Splitting**: Break the document into individual sentences.
2.  **Embedding**: Calculate the vector embedding for every sentence ($S_1, S_2, ... S_n$).
3.  **Similarity Check**: exact sequential comparison. Calculate cosine similarity between $S_i$ and $S_{i+1}$.
4.  **Thresholding**:
    *   Plot the similarities as a graph.
    *   Identify "valleys" (sudden drops in similarity) which represent a change in topic.
    *   Split the chunk at these valleys.

#### Pros & Cons
*   **Pros**: High coherence. No "mid-sentence" cuts. Excellent for messy transcripts.
*   **Cons**:
    *   **Latency**: Requires $N$ embedding calls before you even start indexing.
    *   **Noise**: Single outlier sentences can trigger premature splits.
*   **Production Readiness**: **Medium**. Use for offline indexing pipelines where speed is not critical.

### 5. Parent Document Retrieval (Small-to-Big)
The Gold Standard for production. It decouples the **Indexing Unit** (what you search) from the **Retrieval Unit** (what you send to the LLM).

#### Architecture
*   **Vector Store**: Contains small, dense chunks (e.g., single sentences). Optimized for high-precision search.
*   **Doc Store (Key-Value)**: Contains the original larger documents or windows.

#### Two Main Flavors
1.  **Full Parent Retrieval**:
    *   You stick a "Parent ID" on every small chunk.
    *   When a small chunk is retrieved, you fetch the *entire* parent document (or a large 500-token window) from the Doc Store.
2.  **Sentence Window Retrieval**:
    *   You index a single sentence.
    *   Upon retrieval, you fetch a pre-calculated window of 5 sentences before and 5 after.

#### Why it Works
Embedding a 500-token paragraph "dilutes" the vector. The vector represents the *average* meaning of the whole paragraph. If the answer is in sentence #3, it might get lost.
By embedding sentence #3 directly, you get a sharp vector match. By returning the whole paragraph, you give the LLM the context it needs to reason.

*   **Production Readiness**: **Very High**. State of the art for minimizing hallucinations.

## Comparison & Decision Framework

### Strategy vs Trade-offs

| Strategy          | Computational Cost | Semantic Preservation | Indexing Speed | Best Use Case                          |
| :---------------- | :----------------- | :-------------------- | :------------- | :------------------------------------- |
| **Fixed-Size**    | Lowest             | Poor                  | Fastest        | MVP, Uniform raw text                  |
| **Recursive**     | Low                | Good                  | Fast           | General Purpose Documents              |
| **Markdown/Code** | Low                | Excellent             | Fast           | Technical Documentation, Codebases     |
| **Semantic**      | High               | Excellent             | Slow           | Noisy, unstructured essays/transcripts |
| **Parent Doc**    | Medium             | Excellent             | Medium         | High-accuracy Production RAG           |

### Decision Matrix
*   **Q1: Is the data structured (Code/Markdown/JSON)?**
    *   **Yes** $\rightarrow$ Use **Document Specific** splitters.
*   **Q2: Is high accuracy critical and storage cheap?**
    *   **Yes** $\rightarrow$ Use **Parent Document Retrieval** (Small-to-Big).
*   **Q3: Is the text a stream of consciousness (e.g., meeting transcript) with no structure?**
    *   **Yes** $\rightarrow$ Use **Semantic Chunking**.
*   **Q4: Default fallback?**
    *   $\rightarrow$ Use **Recursive Character Chunking** (Chunk size 1024, Overlap 200).

## Resources
*   **LangChain**: [Text Splitters Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
*   **LlamaIndex**: [Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
*   **Article**: [5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
*   **Visualizer**: [LangChain Chunk Visualizer](https://chunkviz.up.railway.app/)
 ---
**Back to**: [[RAG Index]]
