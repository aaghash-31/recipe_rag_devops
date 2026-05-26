# 🍳 Recipe & Diet Assistant  
## RAG + DevOps Deployment using Docker, Kubernetes & Jenkins

An AI-powered Recipe & Diet Assistant built using Retrieval-Augmented Generation (RAG) and deployed using a complete DevOps pipeline with Docker, Kubernetes, Jenkins, Prometheus, and Grafana.

---

# 📌 Project Overview

This project is a conversational AI assistant capable of:
- Providing recipe recommendations
- Suggesting diet plans
- Answering food and nutrition-related queries
- Generating context-aware responses using RAG architecture

---

# 🚀 Features

## AI Features
- Retrieval-Augmented Generation (RAG)
- Semantic recipe retrieval
- FAISS vector database
- Conversational query handling
- Context-aware AI responses

## DevOps Features
- Dockerized frontend & backend
- Kubernetes deployments
- CI/CD using Jenkins
- Docker Hub integration
- Monitoring using Prometheus & Grafana
- Automated deployments
- Self-healing Kubernetes pods

---

# 🧠 RAG Architecture

```text
User Query
    ↓
Frontend (Streamlit)
    ↓
FastAPI Backend
    ↓
Embedding Model
    ↓
FAISS Vector Search
    ↓
Relevant Context Retrieval
    ↓
Groq LLM
    ↓
Generated Response
```

---

# ⚙️ DevOps Workflow

```text
Developer
   ↓
Git
   ↓
GitHub Repository
   ↓
Jenkins CI/CD Pipeline
   ↓
Docker Image Build
   ↓
Docker Hub Push
   ↓
Kubernetes Deployment
   ↓
Frontend & Backend Pods
   ↓
Monitoring using Prometheus & Grafana
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| AI Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | HuggingFace Embeddings |
| LLM | Groq API |
| Containerization | Docker |
| Container Registry | Docker Hub |
| Orchestration | Kubernetes |
| CLI Management | kubectl |
| CI/CD | Jenkins |
| Monitoring | Prometheus |
| Visualization | Grafana |
| Version Control | Git |
| Repository Hosting | GitHub |
| Cloud Platform | AWS EC2 |

---

# 📂 Project Structure

```text
recipe_rag_devops/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   ├── data_loader/
│   │   └── load_kaggle_recipe_nlg.py
│   ├── rag/
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── __init__.py
│   └── requirements.txt
│
├── frontend/
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
│
├── k8s/
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── jenkins-deployment.yaml
│   ├── jenkins-rbac.yaml
│   ├── jenkins-volume.yaml
│   └── persistent-volume.yaml
│
├── scripts/
├── .gitignore
├── docker-compose.yaml
└── README.md
```

---

# 🐳 Docker Setup

```bash
docker build -t recipe-backend ./backend
docker build -t recipe-frontend ./frontend
```

---

# ☸️ Kubernetes Deployment

```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl get pods
kubectl get svc
```

---

# 🔄 Jenkins CI/CD Pipeline

The Jenkins pipeline automates:
1. Pulling latest source code
2. Building Docker images
3. Pushing images to Docker Hub
4. Updating Kubernetes deployments
5. Restarting application pods

---

# 📊 Monitoring Stack

## Prometheus
Prometheus collects Kubernetes and infrastructure metrics.

## Grafana
Grafana visualizes metrics through dashboards.

---

# 📚 Key Learnings

- Docker containerization
- Kubernetes orchestration
- Jenkins CI/CD automation
- Monitoring setup
- Infrastructure debugging
- Cloud-native deployment workflow

---

# 👨‍💻 Author

## Aaghash A S 

AI/ML + DevOps Enthusiast
