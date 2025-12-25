# 03 - MLOps & Infrastructure

## Overview
Learn to deploy, monitor, and scale ML systems in production. Master the tools and practices for reliable, efficient machine learning operations.

---

## 🔄 Model Lifecycle Management

### Version Control
- [ ] [[ML Version Control Fundamentals]]
- [ ] [[Model Versioning]]
- [ ] [[Dataset Versioning]]
- [ ] [[Experiment Tracking]]
- [ ] [[DVC (Data Version Control)]]

### Experiment Management
- [ ] [[Experiment Tracking Best Practices]]
- [ ] [[MLflow]]
- [ ] [[Weights & Biases]]
- [ ] [[Neptune.ai]]
- [ ] [[Reproducibility in ML]]
- [ ] [[Hyperparameter Tracking]]

### Model Registry
- [ ] [[Model Registry Concepts]]
- [ ] [[Model Metadata Management]]
- [ ] [[Model Lineage Tracking]]
- [ ] [[Model Governance]]

### Deployment Pipelines
- [ ] [[ML Deployment Patterns]]
- [ ] [[CI CD for ML]]
- [ ] [[Shadow Deployment]]
- [ ] [[Rolling Deployment]]

### Monitoring & Observability
- [ ] [[ML Monitoring Fundamentals]]
- [ ] [[Model Performance Monitoring]]
- [x] [[Data Drift Detection]]
- [x] [[Concept Drift]]
- [x] [[Feature Drift]]
- [ ] [[Model Staleness]]
- [ ] [[Logging for ML Systems]]
- [ ] [[Alerting and Incident Response]]

---

## 🏗️ Infrastructure & Scaling

### Compute Resources
- [ ] [[GPU Fundamentals for ML]]
- [ ] [[TPU Overview]]
- [ ] [[GPU Utilization Optimization]]
- [ ] [[Multi-GPU Training]]
- [ ] [[GPU Memory Management]]
- [ ] [[Mixed Precision Training]]

### Distributed Training
- [ ] [[Distributed Training Overview]]
- [ ] [[Data Parallelism]]
- [ ] [[Model Parallelism]]
- [ ] [[Pipeline Parallelism]]
- [ ] [[Distributed Data Parallel (DDP)]]
- [ ] [[Fully Sharded Data Parallel (FSDP)]]
- [ ] [[Horovod]]
- [ ] [[DeepSpeed]]

### Model Optimization
- [ ] [[Model Compression Techniques]]
- [ ] [[Quantization]]
- [ ] [[Pruning]]
- [ ] [[Knowledge Distillation]]
- [ ] [[ONNX Runtime]]
- [ ] [[TensorRT]]
- [ ] [[Neural Architecture Search]]

### Edge Deployment
- [ ] [[Edge ML Fundamentals]]
- [ ] [[TensorFlow Lite]]
- [ ] [[ONNX]]
- [ ] [[CoreML]]
- [ ] [[Model Optimization for Edge]]
- [ ] [[On-Device Inference]]

### Cloud Platforms
- [ ] [[Cloud ML Platforms Overview]]
- [ ] [[AWS SageMaker]]
- [ ] [[Azure Machine Learning]]
- [ ] [[Google Cloud Vertex AI]]
- [ ] [[Databricks ML]]
- [ ] [[Serverless ML Deployment]]

### Container & Orchestration
- [ ] [[Docker for ML]]
- [ ] [[Kubernetes for ML]]
- [ ] [[Kubeflow]]
- [ ] [[MLOps Pipeline Architecture]]
- [ ] [[Model Serving with Kubernetes]]

### Storage & Databases
- [ ] [[ML Storage Strategies]]
- [ ] [[Feature Stores]]
- [ ] [[Object Storage for ML]]
- [ ] [[Model Artifact Storage]]

---

## 📊 Progress Tracking

```dataview
TABLE
  status as "Status",
  difficulty as "Difficulty",
  last_modified as "Last Updated"
FROM "01 - ML & AI Concepts/03 - MLOps & Infrastructure"
WHERE contains(tags, "concept")
SORT file.name ASC
```

---

## 🎓 Learning Path

**Recommended Order:**
1. Start with Model Lifecycle basics (Version Control, Experiment Tracking)
2. Learn Deployment Patterns
3. Study Monitoring & Observability
4. Understand Compute Resources
5. Master Model Optimization
6. Explore Cloud Platforms
7. Advanced: Distributed Training and Edge Deployment

---

**Back to**: [[ML & AI Index]]
