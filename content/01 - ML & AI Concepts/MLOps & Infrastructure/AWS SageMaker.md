# AWS SageMaker

## Overview
AWS SageMaker is a fully managed service that removes the heavy lifting from each step of the machine learning process. It provides a unified toolset to build, train, and deploy ML models at scale. Think of it as "ML in a Box" or a managed interface that abstracts away the underlying EC2 infrastructure, allowing you to focus on the model logic rather than cluster management.

## Key Ideas / Intuition
- **ML in a Box**: SageMaker wraps the entire ML lifecycle (labeling, building, training, tuning, deploying) into a cohesive service. You don't manage servers; you manage "Jobs" and "Endpoints".
- **Separation of Concerns**: In SageMaker, your development environment (Notebook), your training environment (Cluster), and your hosting environment (Endpoint) are completely decoupled. 
    - You write code in a small, cheap notebook instance.
    - You spin up a massive, expensive cluster *only* for the duration of training.
    - You deploy to a right-sized instance for inference.
- **Docker is King**: Under the hood, **everything** in SageMaker is a Docker container. 
    - **Development**: Runs in a container.
    - **Training**: SageMaker spins up EC2, pulls a training image, injects your code/data, trains, saves artifacts to S3, and kills the instance.
    - **Inference**: SageMaker spins up an endpoint, pulls an inference image, downloads your model from S3, and exposes a REST API.

![SageMaker Architecture](https://miro.medium.com/v2/resize:fit:1400/1*Qy60I_tWqMhW5s-c-8s0uQ.png)
*(Conceptual view of the SageMaker workflow: S3 -> Training Container -> S3 -> Inference Container)*

## Core Components (Deep Dive)
### 1. SageMaker Studio
The IDE for ML. It provides a JupyterLab-based interface where you can manage all your SageMaker resources (experiments, pipelines, models, endpoints) visually.

### 2. Processing Jobs
Used for data pre-processing (or post-processing) before training.
- Runs scripts (Python, PySpark) on managed infrastructure.
- Useful for feature engineering and dataset splitting.

### 3. Training Jobs
The workhorse of SageMaker. 
- **Ephemeral Clusters**: You define `InstanceType` (e.g., `ml.p3.2xlarge`) and `InstanceCount`. SageMaker provisions them, runs your job, and terminates them immediately after.
- **Input/Output**: Data streams from S3 (or EFS/FSx), and model artifacts (`model.tar.gz`) are saved back to S3.
- **Distributed Training**: SageMaker handles the complexity of inter-node communication for multi-node training.

### 4. Inference
- **Real-time Inference**: Persistent HTTPS endpoint. Good for low latency requirements. Supports auto-scaling.
- **Serverless Inference**: On-demand usage. Good for intermittent traffic. No idle server costs.
- **Batch Transform**: Offline processing of large datasets. Spins up a cluster, processes all files in S3, saves results, and shuts down.
- **Asynchronous Inference**: Queues requests for large payloads (up to 1GB) or long processing times.

### 5. Model Registry & Pipelines
- **Pipelines**: CI/CD for ML. A DAG (Directed Acyclic Graph) of steps (Processing -> Training -> Evaluation -> Register).
- **Model Registry**: A catalog for your model versions, managing approval status (Pending -> Approved -> Rejected) for deployment.

## Technical Details: The Container Interface
When SageMaker runs your container, it expects a specific directory structure.

### Training Directory Structure (`/opt/ml`)
```bash
/opt/ml
├── input
│   ├── config
│   │   ├── hyperparameters.json # Arguments passed to estimator
│   │   └── resourceConfig.json  # Network info for distributed training
│   └── data
│       └── <channel_name>       # e.g., /train, /validation (Data from S3)
├── model                        # Your script MUST save the model here
│   └── <model files>
├── output
│   └── failure                  # Write failure causes here
└── code                         # Your training script (if using script mode)
    └── train.py
```

### How Execution Works
- **Training**: SageMaker runs `docker run <image> train`. Your container must have an executable named `train`.
- **Inference**: SageMaker runs `docker run <image> serve`. Your container must have an executable named `serve` (usually via a web server like Gunicorn/NGINX).

## Practical Application
### When to Use
- **Production Scale**: You need robust, scalable infrastructure without hiring a DevOps team.
- **Standardization**: You want a unified platform for a large team of data scientists.
- **Governance**: You need lineage tracking (who trained what, on which data) for regulatory reasons.

### When NOT to Use
- **Learning/Hobby**: Can be expensive if you leave endpoints running.
- **Simple Prototyping**: A local GPU or Colab is faster for quick "hello world" experiments.
- **Custom Orchestration**: If you have a highly specialized K8s setup (Kubeflow) and a strong DevOps team, SageMaker's abstractions might feel limiting.

### Cost & Pricing
- **Pay-as-you-go**: Charged per second of instance usage.
- **Endpoints are expensive**: Real-time endpoints run 24/7. **Remember to delete them!**
- **Savings Plans**: Commit to usage for 1-3 years for significant discounts.

## Comparisons

| Feature | AWS SageMaker | EC2 Deep Learning AMI (DLAMI) | Google Vertex AI |
| :--- | :--- | :--- | :--- |
| **Type** | Managed Service | IaaS (Infrastructure as a Service) | Managed Service |
| **Control** | High (Containers) | Max (Root access to OS) | High (AutoML focus) |
| **Ops Overhead** | Low (Managed Infra) | High (Manual Scaling/Patching) | Low |
| **Pricing** | Premium for management | Raw compute cost | Comparable to SageMaker |
| **Hardware** | GPUs (P, G, Trn1, Inf1) | All EC2 Instances | GPUs + **TPUs** |
| **Best For** | Enterprise ML Teams | Researchers / Custom Infra Needs | Google Cloud Native Teams |

## Resources
- **Documentation**: [AWS SageMaker Official Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- **Paper**: [Amazon SageMaker: The Full-Managed Machine Learning Platform](https://arxiv.org/abs/2101.07174) (Not a paper per se, but good architectural overviews exist).
- **Video**: [AWS re:Invent - Deep Dive on Amazon SageMaker](https://www.youtube.com/watch?v=uQc8Itd4UTs)
- **Code**: [SageMaker Examples GitHub](https://github.com/aws/amazon-sagemaker-examples)

## Personal Notes
*   The "Script Mode" in SageMaker is the sweet spot. You use pre-built AWS containers (PyTorch/TensorFlow) but simply pass your own `train.py`. You don't need to build Dockerfiles from scratch 90% of the time.
*   Debugging SageMaker training jobs can be slow because potential failures happen after the instance spin-up (3-5 mins). Test locally first!

## Progress Checklist
- [x] Read overview
- [ ] Understand key concepts (Containers, separation of concerns)
- [ ] Review math (if applicable)
- [ ] Hands-on practice (Run a training job, deploy an endpoint)
- [ ] Can explain to others

**Back to**: [[MLOps & Infrastructure]]
