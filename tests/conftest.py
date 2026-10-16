"""
Shared test fixtures.

Everything runs on THROWAWAY storage (temporary Qdrant folder, registry and trace
database) and on FAKE Groq / Cohere clients, so the tests need no API key, no
network and never touch your real data. The rest of the code under test is the
real project code (real Qdrant in local mode, real BM25, real pipeline).

The environment variables are set before anything imports app.config, which is why
this happens at the top of the file.
"""
import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rag_tests_"))
os.environ["QDRANT_PATH"] = str(_TMP / "qdrant")
os.environ["REGISTRY_PATH"] = str(_TMP / "registry.json")
os.environ["TRACES_DIR"] = str(_TMP / "traces")
os.environ["API_ACCESS_TOKEN"] = "test-api-token"
os.environ["COHERE_API_KEY"] = "test-key"
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["JUDGE_MODEL"] = "test-judge-model"
os.environ["JUDGE_DELAY_SECONDS"] = "0"

from langchain_core.documents import Document  # noqa: E402

from app.config import REFUSAL_MESSAGE  # noqa: E402

# ---------------------------------------------------------------------------
# Fake LLM (Groq)
# ---------------------------------------------------------------------------

_KINDS = [
    ("checking whether an ANSWER is supported", "faithfulness"),
    ("grading an ANSWER against a REFERENCE", "correctness"),
    ("decline to answer because", "refusal"),
    ("checking retrieved text for a question-answering", "sufficiency_or_crag"),
    ("You are a grounded RAG assistant", "generate"),
    ("shortening retrieved text", "compress"),
]


class FakeLLM:
    """Scripted replies. Default behaviour:
      generate     answers with the whole context (so the answer contains the
                   evidence exactly when the evidence reached the LLM)
      correctness  CORRECT if every word of the reference is in the answer
      faithfulness FAITHFUL      refusal  NO
    Override with .by_kind[kind] = text or a function(prompt) -> text, or push one-off
    replies on .queue (used first). Set .fail = True to make every call raise."""

    def __init__(self):
        self.calls = 0
        self.prompts = []
        self.models = []
        self.queue = []
        self.by_kind = {}
        self.fail = False

    def kind_of(self, prompt):
        return next((kind for marker, kind in _KINDS if marker in prompt), "other")

    def reply(self, prompt):
        kind = self.kind_of(prompt)

        if self.queue:
            return self.queue.pop(0)

        custom = self.by_kind.get(kind)

        if custom is not None:
            return custom(prompt) if callable(custom) else custom

        if kind == "generate":
            match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.S)
            return (match.group(1) if match else "").strip() or REFUSAL_MESSAGE

        if kind == "correctness":
            reference = re.search(r"Reference answer:\n(.*?)\n\nAnswer to grade:", prompt, re.S).group(1).lower()
            answer = prompt.split("Answer to grade:\n", 1)[1].split("\n\nReply in exactly")[0].lower()
            ok = all(word in answer for word in re.findall(r"\w+", reference))
            return f"VERDICT: {'CORRECT' if ok else 'INCORRECT'}\nREASON: scripted"

        if kind == "faithfulness":
            return "CLAIMS: 1\nSUPPORTED: 1\nVERDICT: FAITHFUL\nREASON: scripted"

        if kind == "refusal":
            return "REFUSED: NO"

        return "VERDICT: GOOD\nREASON: scripted"


@pytest.fixture
def llm(monkeypatch):
    state = FakeLLM()

    class FakeChatGroq:
        def __init__(self, model, temperature, groq_api_key):
            self.model_name = model

        def invoke(self, prompt):
            state.calls += 1
            state.prompts.append(prompt)
            state.models.append(self.model_name)

            if state.fail:
                raise Exception("Connection error: scripted failure")

            return SimpleNamespace(
                content=state.reply(prompt),
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                response_metadata={},
            )

    import app.generation.llm as llm_module

    monkeypatch.setattr(llm_module, "ChatGroq", FakeChatGroq)
    llm_module.create_llm.cache_clear()
    yield state
    llm_module.create_llm.cache_clear()


# ---------------------------------------------------------------------------
# Fake Cohere (embeddings + rerank)
# ---------------------------------------------------------------------------

class FakeEmbeddings:
    def __init__(self, model=None, cohere_api_key=None):
        pass

    @staticmethod
    def _vector(text):
        out, i = [], 0

        while len(out) < 1024:
            out.extend((b - 128) / 128 for b in hashlib.md5(f"{text}|{i}".encode()).digest())
            i += 1

        return out[:1024]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def cohere(monkeypatch):
    """Fake embeddings and a fake reranker that scores by word overlap."""
    state = SimpleNamespace(rerank_calls=0, fail_rerank=False)

    import cohere as cohere_module
    import app.embeddings.service as service
    import app.retrieval.reranker as reranker

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, model, query, documents, top_n):
            state.rerank_calls += 1

            if state.fail_rerank:
                raise Exception("Cohere says: invalid api key")

            words = set(re.findall(r"\w+", query.lower()))
            scored = sorted(
                ((i, len(words & set(re.findall(r"\w+", d.lower()))) / max(len(words), 1)) for i, d in enumerate(documents)),
                key=lambda pair: -pair[1],
            )

            return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in scored[:top_n]])

    monkeypatch.setattr(cohere_module, "Client", FakeClient)
    monkeypatch.setattr(service, "CohereEmbeddings", FakeEmbeddings)
    service.create_embedding_model.cache_clear()
    service._embed_query_cached.cache_clear()
    reranker._get_client.cache_clear()
    yield state
    service.create_embedding_model.cache_clear()
    service._embed_query_cached.cache_clear()
    reranker._get_client.cache_clear()


# ---------------------------------------------------------------------------
# A small corpus in a REAL (local) Qdrant collection, and a matching dataset
# ---------------------------------------------------------------------------

NAMES = ["zephyr", "quasar", "nimbus", "obsidian", "helios", "tundra", "cobalt", "ember", "lumen", "vortex"]
THINGS = ["quartz", "graphite", "ceramic", "copper", "silica", "carbon", "nickel", "titanium", "cesium", "argon"]


def fact_text(i):
    return f"The {NAMES[i]} module uses {THINGS[i]} filters for signal cleaning. Extra detail sentence number {i}."


def build_collection(name, texts_and_pages, doc_id="doc-test"):
    """Store chunks in a real local Qdrant collection (fake embeddings)."""
    from app.embeddings.service import embed_chunks
    from app.retrieval.bm25 import clear_bm25_cache
    from app.vectorstore.qdrant_store import get_qdrant_client, recreate_collection, store_chunks

    docs = [
        Document(
            page_content=text,
            metadata={"source": "test.pdf", "page": page - 1, "page_number": page, "doc_id": doc_id,
                      "chunk_id": index, "chunk_uid": f"{doc_id}:{name}:{index}", "chunk_strategy": "test"},
        )
        for index, (text, page) in enumerate(texts_and_pages)
    ]
    client = get_qdrant_client()
    recreate_collection(client, name)
    store_chunks(client, docs, embed_chunks(docs), name)
    clear_bm25_cache()

    return docs


def build_dataset(doc_id="doc-test", reviewed=False):
    from app.evaluation.dataset import Dataset, Evidence, Question

    questions = []

    for i in range(10):
        questions.append(Question(
            id=i + 1, question=f"Which filters does the {NAMES[i]} module use?", type="single_hop", expected="answerable",
            ground_truth=f"{THINGS[i]} filters",
            evidence=[Evidence(1, i + 1, f"The {NAMES[i]} module uses {THINGS[i]} filters")],
            tags=["even" if i % 2 == 0 else "odd"], reviewed=reviewed,
        ))

    questions.append(Question(id=11, question="Who won the World Cup in 2022?", type="unanswerable", expected="unanswerable", tags=["off_topic"], reviewed=reviewed))
    questions.append(Question(id=12, question="What does the zephyr module cost?", type="unanswerable", expected="unanswerable", tags=["related_absent"], reviewed=reviewed))

    return Dataset(questions, version=2, source="test.pdf", doc_id=doc_id, pages=14, description="test dataset")


@pytest.fixture
def env(llm, cohere):
    """A ready environment: collection of 10 fact chunks + 4 fillers, matching
    dataset, fake LLM and Cohere. env.config(...) builds a config dict that points
    at this collection (BM25 by default: deterministic)."""
    name = f"col_{uuid.uuid4().hex[:8]}"
    texts = [(fact_text(i), i + 1) for i in range(10)] + [(f"Unrelated filler paragraph number {j} about nothing in particular.", 11 + j) for j in range(4)]
    docs = build_collection(name, texts)

    def config(**changes):
        return {"collection_name": name, "retrieval_method": "bm25", **changes}

    return SimpleNamespace(name=name, docs=docs, dataset=build_dataset(), config=config, llm=llm, cohere=cohere)


@pytest.fixture
def judge(llm):
    from app.evaluation.judges import Judge, JudgeCache

    return Judge(cache=JudgeCache(), delay=0)


@pytest.fixture
def make_collection(cohere):
    """make_collection([(text, page), ...]) -> collection name (real local Qdrant)."""

    def make(texts_and_pages, doc_id="doc-test"):
        name = f"col_{uuid.uuid4().hex[:8]}"
        build_collection(name, texts_and_pages, doc_id)

        return name

    return make


@pytest.fixture(autouse=True)
def fresh_trace_store(tmp_path, monkeypatch):
    """Every test gets its own empty trace database (run ids like run_001 repeat across tests)."""
    import app.tracing.trace_store as trace_store

    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path / "traces")
    trace_store.get_store.cache_clear()
    yield
    trace_store.get_store.cache_clear()