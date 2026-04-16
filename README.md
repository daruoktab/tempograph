# Tempograph

> **Agentic RAG System with Temporal Knowledge Graph for Indonesian Chatbot**
>
> Sistem RAG Agentic dengan Temporal Knowledge Graph untuk chatbot berbahasa Indonesia yang mampu mengingat dan melacak konteks percakapan jangka panjang secara kronologis.

---

## 🎯 Overview

Proyek ini terdiri dari **dua komponen utama**:

### 1. Dataset Generation
Generator dataset percakapan bahasa Indonesia dengan:
- **Causal event graph**: Timeline kehidupan user dengan relasi sebab-akibat antar event
- **Ground truth annotations**: Fakta, entitas, dan referensi temporal dari setiap percakapan
- **Evaluation queries**: Pertanyaan test untuk mengukur performa RAG system

### 2. Agentic Temporal RAG System
Sistem RAG dengan arsitektur agentic yang mengintegrasikan:
- **Temporal Knowledge Graph (TKG)**: Memori terstruktur dengan validitas waktu eksplisit
- **Agentic Retrieval (ReAct)**: Navigasi cerdas pada memori dengan tool-calling
- **Multi-Model Tiering**: Optimasi biaya dengan Gemini Pro/Flash/Lite

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Agentic Retrieval (ReAct)                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ Analyze │ -> │  Plan   │ -> │ Execute │ -> Loop         │
│  └─────────┘    └─────────┘    └─────────┘                 │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Vector    │ │  Temporal   │ │   Entity    │
    │   Search    │ │   Filter    │ │    Lookup   │
    │  (ChromaDB) │ │   (Neo4j)   │ │   (Graph)   │
    └─────────────┘ └─────────────┘ └─────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Temporal Knowledge Graph (Neo4j)               │
│  Nodes: Entity, Episode, Fact                               │
│  Edges: RELATES_TO, CAUSED_BY, VALID_FROM, VALID_TO        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
your-repo/
├── src/
│   ├── config/           # Configuration management
│   ├── dataset/          # Dataset generation module
│   ├── rag/              # RAG system core
│   │   ├── ingestion/    # Data ingestion to TKG
│   │   ├── retrieval/    # Retrieval strategies
│   │   └── vectordb/     # Vector database client
│   ├── llm/              # LLM provider abstraction
│   ├── embedders/        # Embedding models
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utilities
├── scripts/              # CLI scripts
├── examples/             # Example configs & data
├── data/                 # Dataset storage
├── output/               # Evaluation outputs
└── tests/                # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository (ganti URL sesuai remote kamu)
git clone https://github.com/<username>/<repo>.git
cd <repo>

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Generate Dataset

```bash
python scripts/generate_dataset.py \
    --output ./data/dataset \
    --sessions 10 \
    --events 20 \
    --days 60
```

### Run RAG System

```bash
# Ingest data to TKG
python scripts/ingest_data.py --input ./data/dataset

# Evaluate
python scripts/evaluate.py --setup agentic --queries 100
```

## ⚙️ Configuration

### Environment Variables

```bash
# .env
GEMINI_API_KEY=your-gemini-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### Model Configuration

See `src/config/settings.py` for model configurations including:
- Rate limits per model
- Token budgets
- Retrieval parameters

## 📊 Evaluation

The system is evaluated using:
- **Factual Recall**: Accuracy of retrieved facts
- **Temporal Reasoning**: Correct temporal ordering and filtering
- **Multi-hop Reasoning**: Connecting information across sessions
- **LLM Judge**: GPT-based answer quality assessment

## 📚 References

- Adapted from [LOCOMO Framework](https://github.com/ServiceNow/LOCOMO)
- Uses [Graphiti](https://github.com/getzep/graphiti) for TKG management
- Powered by Google Gemini 2.5 models

## 📄 License

MIT License
