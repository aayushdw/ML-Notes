# 05 - Data Engineering

## Overview
Master data pipelines, infrastructure, and management practices essential for ML systems. Learn to build robust data foundations for AI applications.

---

## 💾 Data Management

### Data Versioning
- [ ] [[Data Versioning Fundamentals]]
- [ ] [[Dataset Versioning Strategies]]
- [ ] [[Data Lineage Tracking]]
- [ ] [[Data Provenance]]
- [ ] [[Immutable Data Stores]]

### Feature Engineering
- [ ] [[Feature Engineering for ML]]
- [ ] [[Predictive Power]] - Pearson Correlation, Adjusted Mutual Information, Shapely Value
- [ ] [[Feature Transformation]]
- [ ] [[Feature Encoding]]
- [ ] [[Feature Extraction]]
- [ ] [[Automated Feature Engineering]]
- [ ] [[Feature Crosses]]

### Feature Stores
- [ ] [[Feature Store Fundamentals]]
- [ ] [[Feast]]
- [ ] [[Tecton]]
- [ ] [[Online vs Offline Features]]
- [ ] [[Feature Serving]]
- [ ] [[Feature Reuse]]

### Data Quality
- [ ] [[Data Quality Fundamentals]]
- [ ] [[Data Validation]]
- [ ] [[Data Profiling]]
- [ ] [[Anomaly Detection in Data]]
- [ ] [[Data Cleansing]]
- [ ] [[Missing Data Handling]]

### Data Privacy
- [ ] [[Privacy-Preserving ML]]
- [ ] [[Differential Privacy]]
- [ ] [[Federated Learning]]
- [ ] [[Secure Multi-Party Computation]]
- [ ] [[Homomorphic Encryption]]
- [ ] [[Data Anonymization]]
- [ ] [[GDPR Compliance for ML]]

### Handling Imbalanced Data
- [ ] [[Imbalanced Datasets]]
- [ ] [[Oversampling Techniques]]
- [ ] [[Undersampling Techniques]]
- [ ] [[SMOTE]]
- [ ] [[Class Weights]]
- [ ] [[Anomaly Detection Approaches]]

---

## 🔧 Data Infrastructure

### ETL ELT Pipelines
- [ ] [[ETL vs ELT]]
- [ ] [[Data Pipeline Design Patterns]]
- [ ] [[Batch Processing]]
- [ ] [[Stream Processing]]
- [ ] [[Apache Spark]]
- [ ] [[Apache Flink]]
- [ ] [[DBT (Data Build Tool)]]

### Stream Processing
- [ ] [[Stream Processing Fundamentals]]
- [ ] [[Real-Time Data Processing]]
- [ ] [[Apache Kafka]]
- [ ] [[Kafka Streams]]
- [ ] [[Event Sourcing]]
- [ ] [[CQRS Pattern]]

### Data Storage
- [ ] [[Data Lake Architecture]]
- [ ] [[Data Warehouse Design]]
- [ ] [[Lakehouse Architecture]]
- [ ] [[Delta Lake]]
- [ ] [[Apache Iceberg]]
- [ ] [[Parquet Format]]
- [ ] [[Columnar Storage]]

### Vector Databases
- [ ] [[Vector Databases Overview]]
- [ ] [[Pinecone]]
- [ ] [[Weaviate]]
- [ ] [[Milvus]]
- [ ] [[Qdrant]]
- [ ] [[Chroma]]
- [ ] [[FAISS]]
- [ ] [[Vector Indexing Strategies]]
- [ ] [[Approximate Nearest Neighbor (ANN)]]

### Data Orchestration
- [ ] [[Data Orchestration Tools]]
- [ ] [[Apache Airflow for Data]]
- [ ] [[Workflow Management]]
- [ ] [[DAG Design]]
- [ ] [[Task Dependencies]]
- [ ] [[Backfilling Data]]

### Data Annotation
- [ ] [[Data Labeling Strategies]]
- [ ] [[Active Learning]]
- [ ] [[Weak Supervision]]
- [ ] [[Label Studio]]
- [ ] [[Snorkel]]
- [ ] [[Human-in-the-Loop]]
- [ ] [[Annotation Quality Control]]

### Data Catalog
- [ ] [[Data Discovery]]
- [ ] [[Metadata Management]]
- [ ] [[Data Governance]]
- [ ] [[Schema Registry]]

---

## 📊 Progress Tracking

```dataview
TABLE
  status as "Status",
  difficulty as "Difficulty",
  last_modified as "Last Updated"
FROM "01 - ML & AI Concepts/05 - Data Engineering"
WHERE contains(tags, "concept")
SORT file.name ASC
```

---

## 🎓 Learning Path

**Recommended Order:**
1. Start with Data Quality and Validation
2. Learn Feature Engineering and Feature Stores
3. Study ETL/ELT Pipelines
4. Understand Stream Processing
5. Master Data Storage solutions
6. Explore Vector Databases for AI
7. Advanced: Privacy-Preserving techniques and Data Orchestration

---

**Back to**: [[ML & AI Index]]
