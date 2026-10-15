# RAG Evaluation & Debugging Platform

A project to understand why a Retrieval-Augmented Generation (RAG) system gets an answer right or wrong.

Instead of looking only at the final response, this platform helps inspect what happens during retrieval, reranking, context preparation, and generation. It also provides evaluation metrics to compare different pipeline configurations.

**Links:** [Live Demo](https://rag-evaluation-platform-jyjremkrn8wlna6zqxayys.streamlit.app/) · [API Docs](https://rag-evaluation-platform.onrender.com/docs) · [GitHub](https://github.com/vartika879/rag-evaluation-platform)

## What I Built

- A configurable RAG pipeline with different retrieval strategies.
- Retrieval evaluation using metrics such as Hit@k, Recall@k, and MRR.
- LLM-based evaluation for answer faithfulness, correctness, and refusal behavior.
- Stage-level traces to inspect retrieved chunks and intermediate pipeline outputs.
- Failure analysis to help identify where a RAG response went wrong.
- Experiments comparing chunking and retrieval configurations.

## Evaluation Results

On the project's 60-question evaluation dataset, the reported results were:

| Configuration | Hit@5 | MRR |
|---|---:|---:|
| Vector baseline | 84.8% | 0.731 |
| Hybrid + Reranking | 97.8% | 0.775 |

The hybrid + reranking configuration improved Hit@5 by **13 percentage points** on this dataset.

However, reranking did not improve every query. A regression was observed on Q4, where a relevant result was demoted. This reinforced an important lesson: retrieval changes should be measured against a baseline rather than assumed to improve the system.

*These results should be interpreted alongside the evaluation settings, valid run artifacts, and judging status.*

## How It Works

The application uses Streamlit for the interface and FastAPI for the backend. The RAG pipeline processes a query, retrieves relevant document chunks, optionally applies additional retrieval stages, and generates a response using an LLM.

The evaluation framework measures retrieval performance and, where valid judge results are available, assesses answer quality. Traces help inspect intermediate outputs when a run does not behave as expected.

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Interface:** Streamlit, Plotly
- **Retrieval:** Vector search, BM25, hybrid search, reranking
- **Embeddings and reranking:** Cohere
- **LLM:** Groq
- **Vector database:** Qdrant Cloud
- **Tracing:** SQLite
- **Testing:** pytest

## Run Locally

### Requirements

- Python 3.12+
- Cohere API key
- Groq API key
- Qdrant Cloud credentials, if using cloud-based retrieval

### 1. Clone the repository

```bash
git clone https://github.com/vartika879/rag-evaluation-platform.git
cd rag-evaluation-platform
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file from the project's template:

```powershell
Copy-Item .env.example .env
```

Add the required credentials and settings. Never commit real API keys.

### 3. Start the API

```bash
uvicorn app.api:app --reload --port 8000
```

Open the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Start the interface

In a second terminal:

```bash
streamlit run app/ui.py
```

Open [http://localhost:8501](http://localhost:8501).

### 5. Run the evaluation

```bash
python scripts/run_eval.py run --preset hybrid_rerank
```

For retrieval-only evaluation:

```bash
python scripts/run_eval.py run --preset vector --no-judge
```

Evaluation presets and comparison commands depend on the options supported by the current implementation.

## What I Learned

- Retrieval quality and answer quality need separate evaluation.
- Reranking can improve some queries while hurting others.
- Aggregate metrics alone do not explain individual failures.
- Invalid or unjudged evaluation results should not be treated as successful answers.
- Reproducible runs and inspectable traces make debugging more useful.

## Future Improvements

- More durable trace storage for deployed environments.
- Broader evaluation datasets and stronger judge calibration.
- Improved reproducibility across pipeline configurations.

## License

MIT

## Author

**Vartika Gupta** · [GitHub](https://github.com/vartika879)