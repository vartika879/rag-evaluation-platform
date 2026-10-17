# 🔬 RAG Evaluation & Debugging Platform

A **RAG evaluation and debugging system** that helps developers understand *why* a retrieval-augmented generation pipeline succeeds or fails, using data instead of guesswork.

**🌐 Live Demo:** [rag-evaluation-platform.streamlit.app](https://rag-evaluation-platform-jyjremkrn8wlna6zqxayys.streamlit.app)  
**🔌 API Docs:** [rag-evaluation-platform.onrender.com/docs](https://rag-evaluation-platform.onrender.com/docs)

---

## 🎯 What It Does

Most RAG systems are black boxes. When an answer is wrong, you don't know whether the **retriever** missed the evidence, the **reranker** demoted it, or the **LLM** hallucinated.

This platform opens the box:

- 🔍 **Visualizes** every stage of the pipeline: rewrite, retrieve, rerank, compress, generate
- 📊 **Evaluates** with retrieval metrics (Hit@k, MRR, nDCG) and LLM-judged faithfulness and correctness
- 🎛️ **Compares** 10 pipeline configurations side by side, with bootstrap confidence intervals
- 🐛 **Diagnoses** failures into 10 categories (`retrieval_miss`, `generation_error`, `hallucination`, ...)
- 🧪 **Experiments** with chunking strategies, retrieval methods, and query transformations
- 🔄 **Handles rate limits** gracefully with multi-key API rotation

---

## 📈 Results

Evaluated on a hand-crafted **60-question golden dataset** (46 answerable + 14 unanswerable), with evidence groups per question.

### Retrieval Quality (full 60-question set)

| Configuration | Hit@5 | MRR | Δ Hit@5 |
|---------------|:-----:|:---:|:-------:|
| Vector (baseline) | 84.8% | 0.731 | n/a |
| **Hybrid + Rerank** | **97.8%** | **0.775** | **+13.0 pts** |

### End-to-End Answer Quality (30-question subset, LLM judge)

| Metric | Value |
|--------|:-----:|
| **Answer Correctness** | **96%** (22/23 answerable) |
| **Refusal Accuracy** | **100%** (7/7 unanswerable) |
| **Hit@5 (this subset)** | 96% |
| **Remaining failures** | 1 partial (known reranker over-correction) |
| **Rate-limit errors** | 0 (multi-key rotation) |

> **Note:** The two tables use different sample sizes (60 vs. 30 questions), so their Hit@5 values are not directly comparable.

**Key findings**
- Hybrid + Rerank recovered several retrieval misses that pure vector search failed on.
- One regression is documented (Q4, reranker over-correction). The tradeoff is shown, not hidden.
- All 7 unanswerable questions in the evaluated subset were correctly refused.

**Caveat:** With only 23 answerable questions in the end-to-end run, small differences (a few percentage points) can be noise. That is why the evaluation framework reports bootstrap confidence intervals.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│  Streamlit UI (7 pages: Overview, Ask, Experiments,  │
│  Evaluation, Debug Traces, Documents, About)         │
└───────────────────────┬──────────────────────────────┘
                        │ HTTP
┌───────────────────────▼──────────────────────────────┐
│  FastAPI Backend  (async)                            │
└───────────────────────┬──────────────────────────────┘
                        │
     ┌──────────────────┼──────────────────┐
     ▼                  ▼                  ▼
┌──────────┐     ┌─────────────┐     ┌──────────┐
│ Pipeline │────▶│  Retrieval  │────▶│  Qdrant  │
│          │     │ (10 presets)│     │  Cloud   │
└────┬─────┘     └─────────────┘     └──────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Evaluation Framework                                │
│ • Custom metrics (bootstrap CI, paired tests)       │
│ • LLM-as-judge (faithfulness, correctness, refusal) │
│ • Failure diagnosis (10 categories)                 │
└─────────────────────────────────────────────────────┘
```

### Pipeline Stages

Every query passes through up to 8 configurable stages:

`rewrite → query_transform → retrieve → rerank → crag → compress → context → generate`

Each stage is **timed**, **traced**, and **saved to SQLite** for debugging.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit + Plotly |
| **Vector DB** | Qdrant Cloud (HNSW, cosine) |
| **Embeddings** | Cohere `embed-english-v3.0` (1024-dim) |
| **Reranker** | Cohere `rerank-v3.5` |
| **LLM** | Groq `openai/gpt-oss-20b` |
| **BM25** | rank-bm25 |
| **Chunking** | LangChain (recursive, fixed, semantic, structure-aware) |
| **Tracing** | SQLite (WAL mode) |
| **Evaluation** | Custom (bootstrap CI, paired significance tests) |
| **Deploy** | Render (API) + Streamlit Cloud (UI) + Qdrant Cloud |

---

## ✨ Features

### 🎛️ 10 Pipeline Presets

Switch between retrieval strategies from the UI:

`vector`, `bm25`, `hybrid`, `hybrid_weighted`, `vector_rerank`, `hybrid_rerank`, `hyde`, `expand`, `multi_query`, `hybrid_rerank_crag`

### 🔬 Stage-by-Stage Breakdown

Every query shows:
- HyDE's generated hypothetical passage
- Multi-query reformulations
- Reranker rank changes (with ↑↓ arrows)
- CRAG grading verdict
- Context compression ratio

### 🔄 Multi-Key Rotation

Groq's free tier has strict daily token limits. The system rotates between multiple API keys:
- 3 retries per key, then automatic rotation
- Thread-safe with locking
- Labels the failure as `provider_rate_limit` if all keys are exhausted

### 🐛 Failure Diagnosis

Every failed question is auto-labeled with one of:

`retrieval_miss`, `lost_before_llm`, `generation_error`, `hallucination`, `false_refusal`, `provider_rate_limit`, `pipeline_error`, `correct_refusal`, `partial`, `ok`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- API keys: [Cohere](https://dashboard.cohere.com/), [Groq](https://console.groq.com/), [Qdrant Cloud](https://cloud.qdrant.io/)

### Installation

```bash
git clone https://github.com/vartika879/rag-evaluation-platform.git
cd rag-evaluation-platform
python -m venv .venv
.venv\Scripts\activate    # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your API keys to .env
```

### Ingest & Run

```bash
# Ingest the bundled PDF
python -m app.ingestion.ingest data/documents/rag_survey.pdf --recreate

# Run the API
uvicorn app.api:app --reload --port 8000

# Run the UI (in a new terminal)
streamlit run app/ui.py
```

Open http://localhost:8501

### Run an Evaluation

```bash
# 30-question run with LLM judge
python scripts/run_eval.py run --preset hybrid_rerank --limit 30

# Compare two runs
python scripts/run_eval.py compare run_010 run_050
```

---

## 🎓 Design Decisions

**Why hybrid search?**  
Vector search can miss exact keyword matches (proper nouns, numbers), while BM25 misses semantic paraphrases. RRF fusion combines both. Measured gain on this dataset: **+13.0 percentage points Hit@5** (with reranking).

**Why custom evaluation instead of RAGAS?**  
Full control over judge prompts, judge caching, and calibration against human labels, with better observability into each failure.

**Why multi-key rotation?**  
A full evaluation run consumes a large share of Groq's free daily token budget. Rotating keys lets a run complete without dropping questions.

**Why SQLite for traces?**  
Zero-config, atomic writes, and WAL mode for concurrent reads. Easily swappable for Postgres in production.

---

## 📚 Learnings

- **Reranker over-correction is real.** In Q4 the correct chunk dropped from rank 7 to 18 because the reranker disagreed with hybrid fusion. It is a case-by-case tradeoff, not a free win.
- **Rate limits need more than retries.** Rotate keys and label the failure so it isn't mistaken for a quality problem.
- **Small datasets need confidence intervals.** With 23 answerable questions, a few points of difference is often noise.

---

## ⚠️ Limitations

- Golden dataset is small (60 questions) and built around a single document (the bundled RAG survey paper).
- End-to-end LLM-judged results come from a 30-question subset, so treat them as indicative, not definitive.
- LLM-as-judge can be biased; calibration against human labels is ongoing.

---

## 📄 License

MIT

---

## 👤 Author

**Vartika**  
[GitHub](https://github.com/vartika879)

*Built as a portfolio project to demonstrate end-to-end RAG system design, evaluation methodology, and deployment.*