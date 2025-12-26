# Google Cloud Vertex AI

## Overview
Vertex AI is Google Cloud's fully managed, unified machine learning platform. It brings together Google's previously separate services (AutoML and AI Platform) into a single console and API. It is designed to accelerate the deployment of AI models by providing tools for every step of the ML workflow, from data engineering to training and MLOps, with a heavy emphasis on "Serverless" experiences and state-of-the-art AutoML capabilities.

## Key Ideas / Intuition
- **Unified Platform**: Vertex AI is not just one tool but a suite. It unifies **AutoML** (for low-code users) and **Custom Training** (for advanced engineers) under the same umbrella.
- **Serverless MLOps**: Unlike SageMaker which feels like "Managed EC2", Vertex AI feels more "Serverless". You worry less about the underlying cluster management and more about the `Job` configuration.
- **The "Model Garden"**: Vertex AI is the home for Google's Foundation Models (Gemini, PaLM, Imagen). It’s not just for training *your* models, it's for consuming *their* models via APIs.

![Vertex AI Workflow](https://storage.googleapis.com/gweb-cloudblog-publish/images/Vertex_AI_diagram_2.max-2000x2000.png)
*(Conceptual flow: Data -> Workbench/AutoML/Custom Job -> Model Registry -> Endpoint)*

## Core Components (Deep Dive)
### 1. Vertex AI Workbench
The development environment.
- **Managed Notebooks**: Fully managed JupyterLab instances. Google handles the OS, drivers (CUDA), and connection to data (BigQuery).
- **Deep Integration**: Can query BigQuery directly from the notebook cell using SQL magic (`%%bigquery`).

### 2. AutoML
Google's flagship offering for those who want SOTA results without writing model code.
- **Supported Types**: Image (Classification, Object Detection), Tabular (Classification, Regression, Forecasting), Text, and Video.
- **Under the hood**: It performs automatic Neural Architecture Search (NAS) and hyperparameter tuning.

### 3. Custom Training Jobs
For when you need full control (PyTorch, TensorFlow, XGBoost).
- **Containerized**: You supply a Docker container (pre-built or custom).
- **Distributed**: Supports reduction servers and multi-node training out of the box.
- **Hyperparameter Tuning (Vizier)**: Uses Google's internal "Vizier" service for black-box optimization.

### 4. Model Garden & Generative AI
- **Model Garden**: A searchable library of foundation models (Gemini, Llama 2, Claude, etc.).
- **Generative AI Studio**: A UI sandbox to prototype prompts, tune models (RLHF, adapter tuning), and deploy them.

### 5. Prediction
- **Online Prediction**: HTTP endpoint for real-time scoring.
- **Batch Prediction**: Process data in GCS/BigQuery and output results to GCS/BigQuery.
- **Explained AI**: Built-in feature attribution (Shapley values) to explain *why* a model made a prediction.

### 6. Vertex AI Pipelines
- Managed service for running **Kubeflow Pipelines (KFP)**.
- **Metadata Store**: Automatically tracks artifacts (datasets, models, metrics) produced by each step.

## Technical Details: Custom Containers
To use a custom container for training or prediction, you must adhere to Google's contract.

### Environment Variables
Vertex AI injects these variables into your container at runtime:
- **`AIP_MODEL_DIR`**: A Cloud Storage URI (e.g., `gs://bucket/job-dir/model/`). Your script **MUST** save the model artifacts here.
- **`AIP_STORAGE_URI`**: (For Prediction) The URI where model artifacts are stored.
- **`AIP_http_port`**: (For Prediction) The port your web server must listen on (default: `8080`).
- **`AIP_DATA_FORMAT`**: Format of the input data (e.g., `jsonl`, `csv`).

### Example Training Snippet
```python
import os

model_dir = os.environ['AIP_MODEL_DIR']
# ... train model ...
model.save(model_dir) # Save directly to the GCS bucket path provided
```

## Practical Application
### When to Use
- **Deep Learning / GenAI Focus**: You want access to TPUs or the latest Foundation Models (Gemini).
- **AutoML Needs**: You have a tabular dataset and want a strong baseline immediately without coding.
- **BigQuery User**: Your data is already in BigQuery; Vertex AI integration is seamless.

### When NOT to Use
- **Simple VM Needs**: If you just want a raw VM with a GPU and no MLOps overhead, a Deep Learning VM (DLVM) on GCE is simpler.
- **AWS ecosystem**: If all your data is in S3, the egress costs and latency to move data to Vertex AI might be prohibitive.

### Cost & Pricing
- **Training**: Charged per **Node Hour** (e.g., 1 hour of `n1-standard-4` + `T4 GPU`).
- **Prediction**:
    - **Custom-trained**: Charged per Node Hour (24/7 if not scaled to zero).
    - **AutoML**: Often carries a premium per node hour.
    - **GenAI**: Charged per **1k characters** (text) or per image.

## Comparisons

| Feature | Vertex AI | AWS SageMaker |
| :--- | :--- | :--- |
| **Philosophy** | Serverless / Automated | Managed Infra / Builder-focused |
| **AutoML** | Class-leading (Tabular/Vision) | Strong (Autopilot) |
| **Notebooks** | Workbench (Deep BQ integration) | Studio (Deep AWS integration) |
| **Hardware** | GPUs + **TPUs** | GPUs + Inferentia/Trainium |
| **GenAI** | Extensive (Gemini, PaLM, Garden) | Bedrock / JumpStart |
| **Orchestration** | Kubeflow Pipelines (Managed) | SageMaker Pipelines |

## Resources
- **Documentation**: [Vertex AI Official Docs](https://cloud.google.com/vertex-ai/docs)
- **Codelabs**: [Vertex AI Codelabs](https://codelabs.developers.google.com/?product=vertex_ai) - Great for step-by-step tutorials.
- **Video**: [Vertex AI: Machine Learning on Google Cloud](https://www.youtube.com/watch?v=gT4p-HCKY9w)
