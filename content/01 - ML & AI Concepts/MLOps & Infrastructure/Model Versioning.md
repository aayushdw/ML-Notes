# Model Versioning

## Overview
Model Versioning in enterprise MLOps is the process of tracking machine learning models not just as files, but as versioned software artifacts with complete lineage and lifecycle management. Unlike simple code versioning, model versioning must capture the **code**, **data**, **configuration**, and **environment** that produced a specific model artifact.

In production systems, this is solved using a **Model Registry**, which acts as a central repository to manage model deployment stages (e.g., Staging, Production) and ensures reproducibility.

## Key Ideas / Intuition
### 1. The Model Registry
A Model Registry is like "Git for binaries + Metadata" or a "Package Manager for ML".

### 2. The "Recipe" (Lineage)
A versioned model is not just a `.pkl` or `.onnx` file. It is a tuple:
$$ \text{Model Version} = f(\text{Data Ver}, \text{Code Ver}, \text{Hyperparams}, \text{Environment}) $$
If *any* of these change, you have a new model version. Enterprise systems automate the tracking of this lineage so you can answer: "Which dataset trained the model running in production right now?"

### 3. Lifecycle & Promotion
Models are not static; they move through stages.
- **Registration**: A training run finishes, and the artifact is saved as `Version 1`.
- **Staging**: `Version 1` is promoted to "Staging" for integration testing.
- **Production**: After passing tests, `Version 1` is promoted to "Production" (replacing `Version 0`).
- **Archived**: `Version 0` is retired but kept for rollback capability.

**Immutable Artifacts**: You never rebuild a model for production. You build *once* (during training), register it, and that *exact same binary* is promoted through environments.

### 4. Decoupling Deployment from Training
By using a registry, inference services don't need to hardcode paths like `s3://bucket/model_v2.pt`. Instead, they query the registry:
> "Give me the latest model marked as 'Production' for 'FraudDetection'."
This allows you to update the model in the background (by promoting a new version) without changing the serving code.

## Practical Application

### Tooling
*   **MLflow Model Registry**: The industry standard for open-source (and Databricks) model management.
*   **AWS SageMaker Model Registry**: Deeply integrated with the AWS ecosystem.
*   **Weights & Biases (WandB)**: Excellent for experiment tracking that transitions into a registry.
*   **Azure ML**: Similar concept to SageMaker.
*   **BentoML**: Focuses on serving but includes a model store.

### Best Practices
*   **One Version, One ID**: Every registered model must have a unique immutable ID.
*   **Automated Registration**: CI/CD pipelines should automatically register models that pass a certain metric threshold during training.
*   **Gated Promotion**: Moving a model from "Staging" to "Production" should require a manual approval or a rigorous automated test suite (Canary evaluation).
*   **Semantic Versioning**: While registries use integers (v1, v2), you should also use tags or descriptions for semantic meaning (e.g., `v1.2.0-bert-large`).

### Trade-offs
*   **Complexity**: Adds a new infrastructure component (the Registry server).
*   **Storage**: Storing every model version can be expensive if models are huge (LLMs). Retention policies are needed.

## Comparisons

| Feature | **Git** | **Model Registry** |
| :--- | :--- | :--- |
| **Primary Unit** | Source Code (Text) | Model Artifacts (Binary) |
| **Versioning Logic** | Diffs / Merges | Immutable Snapshots / Linear History |
| **Metadata** | Commit Messages, Author | Metrics (Accuracy), Hyperparams, Data Lineage |
| **Lifecycle** | Branches (Dev/Main) | Stages (None, Staging, Prod, Archived) |
| **Use Case** | Developing the training script | Managing the deployable asset |

## Resources
*   **Papers/Articles**:
    *   [Google: MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
    *   [Databricks: Managing the Complete Machine Learning Lifecycle with MLflow](https://databricks.com/blog/2019/10/17/managing-the-complete-machine-learning-lifecycle-with-mlflow-model-registry.html)
*   **Docs**:
    *   [MLflow Model Registry Documentation](https://mlflow.org/docs/latest/model-registry.html)
    *   [WandB Model Registry](https://docs.wandb.ai/guides/model_registry)


---
**Back to**: [[00 - MLOps & Infrastructure Index]]
