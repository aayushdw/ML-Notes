
# ML API Design Principles

## Overview
A fundamental principle in ML API design is the strict architectural decoupling of **Inference** (Prediction) from **Training** (Learning). These two workloads have diametrically opposed system requirements.

| **Feature**  | **Inference API**                             | **Training API**                                 |
| ------------ | --------------------------------------------- | ------------------------------------------------ |
| **Latency**  | **Critical** (<100ms usually required).       | **Flexible** (Hours to days).                    |
| **Compute**  | **CPU/Small GPU** (Bursty, high concurrency). | **Heavy GPU/TPU** (Sustained, batch processing). |
| **State**    | **Stateless** (Ideally).                      | **Stateful** (Checkpoints, logs).                |
| **Protocol** | **gRPC / HTTP/2** (Minimize overhead).        | **Async HTTP / Webhooks** (Long-running).        |

### The "Command-Query" Separation for ML

Treat ML system using a variation of **CQRS (Command Query Responsibility Segregation)**:
- **Queries (Inference):** Read-only operations that return predictions. Optimized for **Throughput** ($\lambda$) and **Latency** ($W$).
- **Commands (Training/Fine-tuning):** Write operations that update model state. Optimized for reliability and resource utilization.

## Protocol Selection & Payload Optimization
"JSON over REST" is often insufficient for heavy tensor payloads. Protocols evaluations should be based on serialization overhead and transport efficiency.

### The Protocol Decision Matrix
- **REST (JSON):** Use for low-frequency management APIs (e.g., `list_models`, `update_config`) or public-facing APIs where developer experience (DX) > raw performance.
    
- **gRPC (Protobuf):** The standard for internal service-to-service inference.
    - **Why:** HTTP/2 multiplexing and binary serialization.
    - **Performance:** Benchmarks typically show a **7-10x reduction in latency** compared to REST for payload-heavy requests.

### Binary Serialization Deep Dive

When designing the schema for your `PredictRequest`, the serialization format dictates the "tax" you pay on every call.

1. Protocol Buffers:
    - **Pros:** Strongly typed, backward compatible, excellent tooling (gRPC).
    - **Cons:** Requires a deserialization step (parsing).

2. FlatBuffers (https://en.wikipedia.org/wiki/FlatBuffers):
    - **Mechanism:** Accesses serialized data without parsing/unpacking. It uses offset tables to read data directly from the buffer.
    - **Use Case:** Mobile/Edge ML deployment where CPU cycles for parsing are expensive.
    - **Trade-off:** slightly larger payload size on wire compared to Protobuf, but effectively **zero-latency parsing**.


## Queueing Theory for Inference

To rigorously design for SLA (Service Level Agreement), apply queuing theory. An inference server can be modeled as an $M/M/c$ queue (Markovian arrival, Markovian service times, $c$ servers).
### Little's Law
The fundamental theorem governing your API's concurrency:
$$L = \lambda W$$
Where:
- $L$ = Average number of requests in the system (Concurrency).
- $\lambda$ = Average arrival rate (Requests per second - RPS).
- $W$ = Average time a request spends in the system (Latency).

Design Implication:
If your model takes $W = 0.2s$ (200ms) to infer, and you need to handle $\lambda = 1000$ RPS:
$$L = 1000 \times 0.2 = 200$$
You need system capacity (concurrency) to handle 200 active requests simultaneously. This dictates your GPU memory sizing and worker count.

### The Batching Cost Function

Batching improves throughput but harms latency. We can define a cost function $C(b)$ to find the optimal batch size $b$:
$$C(b) = \alpha \cdot \text{Latency}(b) + \frac{\beta}{\text{Throughput}(b)}$$
Where:
- $\text{Latency}(b) \approx T_{overhead} + b \cdot T_{compute}$ (simplified linear approx).
- $\text{Throughput}(b) \approx \frac{b}{T_{overhead} + b \cdot T_{compute}}$.
- $\alpha, \beta$ are weights based on business priority (e.g., Real-time user vs. Offline job).

**Actionable Insight:** Expose a `max_batch_size` and `batch_timeout` parameter in your API configuration (or dynamic batching sidecar like Triton) to tune this curve.

## Asynchronous Patterns for Long-Running Operations (LROs)
For Generative AI (e.g., image generation) or Batch Processing, a synchronous `200 OK` is an anti-pattern. Use the **Polled Async Request-Reply** pattern.

1. **Client POSTs request:** `POST /v1/jobs/generate`
2. **Server accepts immediately:** Returns `202 Accepted` with a `Location` header pointing to a status endpoint.
```http
HTTP/1.1 202 Accepted
Location: /v1/jobs/12345/status
Retry-After: 5
```
3. **Client Polls:** `GET /v1/jobs/12345/status` returns `{"status": "processing"}`.
4. **Completion:** Eventually returns `303 See Other` (redirect to result) or `200 OK` with the payload.

**Advanced Variation:** Use **Webhooks** for the completion signal to avoid "chatty" polling if the job duration is highly variable (minutes to hours).


## LLM Specifics: Evaluation & Feedback Loops
For LLMs, the API must support the **Data Flywheel**(using user data to create continuous improvement cycle). You are not just serving predictions; you are harvesting data for future fine-tuning (RLHF).
### A. The "Feedback" Endpoint
Every generation endpoint should return a `request_id` or `trace_id`. The API must have a companion endpoint to capture human feedback on that specific trace.

## Resources
### Papers
- 

### Articles & Blog Posts
- 

### Videos & Tutorials
- 

### Code Examples
- 

### Books
- 

## Questions / Further Research
- [ ] 

---

**Progress**: 
- [x] Read overview materials
- [x] Understand key concepts
- [ ] Review mathematical foundations
- [ ] Study implementations
- [ ] Complete hands-on practice
- [x] Can explain to others

**Status Options**: `not-started` | `in-progress` | `completed` | `review-needed`
**Difficulty Options**: `beginner` | `intermediate` | `advanced` | `expert`

---
**Back to**: [[ML & AI Index]]
