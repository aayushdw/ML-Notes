---
tags:
  - index
  - ml-ai
  - llm
  - generative-ai
---

# 02 - LLMs & Generative AI

## Overview
Master Large Language Models and Generative AI systems. Learn how to work with, optimize, and deploy modern language models in production environments.

---

## 🔧 LLM Operations

### Prompt Engineering
- [ ] [[Prompt Engineering Fundamentals]]
- [ ] [[Prompt Design Patterns]]
- [ ] [[Prompt Optimization Techniques]]
- [x] [[Prompt Injection and Safety]]
 
### Fine-Tuning Strategies
- [ ] [[Fine-Tuning Overview]]
- [x] [[Full Fine-Tuning]] (Not needed, Captured in above doc)
- [x] [[LoRA (Low-Rank Adaptation)]]
- [x] [[QLoRA (Quantized LoRA)]]
- [ ] [[P-Tuning, Prefix-Tuning]]
- [x] [[Adapter Methods]](Can be found in FineTuning Overview)

### Model Evaluation
- [ ] [[LLM Evaluation Metrics]]
- [ ] [[Perplexity]]
- [ ] [[BLEU Score]]
- [ ] [[ROUGE Scores]]
- [ ] [[Human Evaluation Methods]]
- [ ] [[LLM Benchmarks (MMLU, HellaSwag, etc.)]]

### Context Management
- [x] [[Context Window Fundamentals]]
- [ ] [[Token Optimization]]
- [ ] [[Context Compression Techniques]]
- [ ] [[Sliding Window Attention]]
- [ ] [[Long-Context Models]]

---

## 🚀 Production LLM Systems

### Model Serving & Inference
- [ ] [[Model Serving Architecture]]
- [ ] [[Inference Optimization]]
- [ ] [[Batching Strategies]]
- [ ] [[KV Cache Optimization]]
- [ ] [[Speculative Decoding]]
- [ ] [[Model Quantization for Inference]]

### Caching & Cost Management
- [ ] [[Semantic Caching]]
- [ ] [[Response Caching Strategies]]
- [ ] [[Cost Optimization Techniques]]
- [ ] [[Request Deduplication]]
- [ ] [[Token Budget Management]]

### Safety & Alignment
- [ ] [[LLM Safety Fundamentals]]
- [ ] [[Content Filtering]]
- [ ] [[Guardrails and Constraints]]
- [ ] [[RLHF (Reinforcement Learning from Human Feedback)]]
- [ ] [[Constitutional AI]]
- [ ] [[Red Teaming LLMs]]

### RAG Systems
- [ ] [[RAG (Retrieval Augmented Generation) Overview]]
- [ ] [[Vector Databases for RAG]]
- [ ] [[Embedding Models]]
- [ ] [[Retrieval Strategies]]
- [ ] [[Hybrid Search (Dense + Sparse)]]
- [ ] [[Reranking Techniques]]
- [ ] [[RAG Evaluation]]

### Multi-Modal Systems
- [ ] [[Multi-Modal Models Overview]]
- [ ] [[Vision-Language Models]]
- [ ] [[Text-to-Image Generation]]
- [ ] [[Image-to-Text Generation]]
- [ ] [[Audio Processing with LLMs]]

### Agent Systems
- [ ] [[LLM Agents Fundamentals]]
- [ ] [[ReAct Pattern]]
- [ ] [[Tool Use and Function Calling]]
- [ ] [[Agent Orchestration]]
- [ ] [[Multi-Agent Systems]]
- [ ] [[Agent Memory Systems]]

### Generative Models
- [ ] [[Transformer Language Models]]
- [ ] [[GPT Architecture]]
- [ ] [[BERT and Encoder Models]]
- [ ] [[T5 and Encoder-Decoder Models]]
- [ ] [[Diffusion Models]]
- [ ] [[GANs (Generative Adversarial Networks)]]
- [ ] [[VAEs (Variational Autoencoders)]]

---

## 📊 Progress Tracking

```dataview
TABLE
  status as "Status",
  difficulty as "Difficulty",
  last_modified as "Last Updated"
FROM "01 - ML & AI Concepts/02 - LLMs & Generative AI"
WHERE contains(tags, "concept")
SORT file.name ASC
```

---

## 🎓 Learning Path

**Recommended Order:**
1. Start with Prompt Engineering basics
2. Understand Context Management
3. Learn Fine-Tuning Strategies
4. Study RAG Systems
5. Explore Production considerations (Serving, Caching, Safety)
6. Advanced: Agent Systems and Multi-Modal

---

**Back to**: [[ML & AI Index]]
