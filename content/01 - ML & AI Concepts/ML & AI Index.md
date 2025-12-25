# 🤖 ML & AI Knowledge Base

## Overview
This is your comprehensive knowledge base for Machine Learning and AI Engineering concepts. This structure follows a senior AI engineer's learning path, covering everything from fundamentals to specialized domains and business strategy.

## 📚 Main Categories

### [[01 - Core Fundamentals Index|01 - Core Fundamentals]]
Build your foundation in ML/AI with essential concepts
- Machine Learning Basics
- Deep Learning Architecture

### [[02 - LLMs & Generative AI Index|02 - LLMs & Generative AI]]
Master modern language models and generative systems
- LLM Operations
- Production Systems

### [[03 - MLOps & Infrastructure Index|03 - MLOps & Infrastructure]]
Learn to deploy, scale, and maintain ML systems in production
- Model Lifecycle Management
- Infrastructure & Scaling

### [[04 - Software Engineering Index|04 - Software Engineering for AI]]
Apply software engineering best practices to AI systems
- System Design
- Code Quality & Testing

### [[05 - Data Engineering Index|05 - Data Engineering]]
Master data pipelines and infrastructure for ML
- Data Management
- Data Infrastructure

### [[06 - Specialized Domains Index|06 - Specialized Domains]]
Deep dive into specific AI application areas
- Computer Vision
- Natural Language Processing
- Ethics & Responsible AI

### [[07 - Business & Strategy Index|07 - Business & Strategy]]
Bridge technical skills with business impact
- Strategic Skills
- Leadership & Communication

---

## 📊 Learning Progress

### Overall Progress by Category
```dataview
TABLE WITHOUT ID
  category as "Category",
  length(rows.file.name) as "Total Topics",
  length(filter(rows.status, (s) => s = "completed")) as "Completed",
  round((length(filter(rows.status, (s) => s = "completed")) / length(rows.file.name)) * 100) + "%" as "Progress"
FROM "01 - ML & AI Concepts"
WHERE contains(tags, "concept")
GROUP BY category
SORT category ASC
```

### Recently Studied
```dataview
TABLE 
  category as "Category",
  status as "Status",
  last_modified as "Last Modified"
FROM "01 - ML & AI Concepts"
WHERE contains(tags, "concept")
SORT last_modified DESC
LIMIT 10
```

### In Progress
```dataview
TABLE 
  category as "Category",
  date_created as "Started"
FROM "01 - ML & AI Concepts"
WHERE contains(tags, "concept") AND status = "in-progress"
SORT date_created DESC
```

### To Study Next
```dataview
TASK
FROM "01 - ML & AI Concepts"
WHERE contains(tags, "index")
```

---

## 🔗 Quick Access

### Current Focus
<!-- Update this section with what you're currently learning -->
- Currently studying: 
- Next up: 

### Important Concepts
<!-- Pin your most important or frequently referenced concepts here -->
- [[Transformers]]
- [[Attention Mechanism]]
- [[RAG (Retrieval Augmented Generation)]]

### Project Connections
- [[Privacy Aware Semantic Caching - Status]]

---

## 📝 Study Notes

### This Week's Goals
- [ ] 
- [ ] 
- [ ] 

### Questions to Explore
- [ ] 
- [ ] 
- [ ] 

---

**Navigation**:
- 🏠 [[HOME]]
- 📊 [[Fitness Dashboard]]
- 🚀 [[Privacy Aware Semantic Caching - Status]]
