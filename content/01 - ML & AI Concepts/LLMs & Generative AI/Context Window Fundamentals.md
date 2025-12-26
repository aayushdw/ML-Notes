## Overview
The context window is the maximum amount of text that an LLM can "see" at once—including both input and output.

### What Consumes Context Window?

Everything that goes into the model counts toward the limit:

1. System Instructions (often invisible to users)
```
"You are Claude, created by Anthropic..."
[Entire system prompt with behavior guidelines]
~ 1,000-3,000 tokens typically
```

2. Conversation History: both user messages and assistant responses
3. Current Message
4. Uploaded Documents
5. Model's Response: (as it's being generated)
6. Tool Use / Function Calls
## Key Ideas
- Just because a model CAN handle 200K tokens doesn't mean it processes all of them equally well. Performance often degrades with very long contexts.

## The "Lost in the Middle" Problem

**Critical Discovery**: Models don't pay equal attention to all parts of the context window.
Research has shown that language models are best at using information that appears:
1. At the very beginning (primacy effect)
2. At the very end (recency effect)

Information in the **middle** of a very long context is often overlooked or forgotten.

## Context Window Management Strategies

### Summarization and Compression
- When conversations get long, summarize earlier exchanges.
- Summarize at conversations exceeding 30-40% of context window.
### Chunking Large Documents - RAG (TODO)


## Code Examples

### Token Counting

#### OpenAI's Tiktoken (Library)
```python
import tiktoken

# For GPT models
encoding = tiktoken.get_encoding("cl100k_base")
text = "Your text here"
tokens = encoding.encode(text)
print(f"Token count: {len(tokens)}")
```

#### Anthropic's Token Counter (API)
````python
import anthropic
client = anthropic.Anthropic(api_key="your-key")
# Count tokens
message = client.messages.count_tokens(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Your text here"}]
)
print(f"Token count: {message.input_tokens}")
````

---
**Back to**: [[ML & AI Index]]
