
from fastapi.testclient import TestClient
from app.api import app
import app.api as api_module

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(monkeypatch):
    def fake_answer_question(question, top_k=5):
        return {
            "question": question,
            "answer": "RAG combines retrieval and generation.",
            "refused": False,
            "sources": [
                {"source": "rag_survey.pdf", "page": 1, "score": 0.8}
            ],
        }

    monkeypatch.setattr(api_module, "answer_question", fake_answer_question)

    response = client.post(
        "/ask",
        json={"question": "What is RAG?", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAG combines retrieval and generation."
    assert data["refused"] is False
    assert data["sources"][0]["page"] == 1


def test_ask_rejects_blank_question():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={"question": "What is RAG?", "top_k": 100},
    )

    assert response.status_code == 422




def test_ask_refuses_when_no_documents_found(monkeypatch):
    def fake_answer_question(question, top_k=5):
        return {
            "question": question,
            "answer": "I don't have enough information in the provided documents.",
            "refused": True,
            "sources": [],
        }

    monkeypatch.setattr(api_module, "answer_question", fake_answer_question)

    response = client.post(
        "/ask",
        json={"question": "Unanswerable question"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is True
    assert data["sources"] == []


def test_ask_returns_llm_refusal(monkeypatch):
    refusal = "I don't have enough information in the provided documents."

    def fake_answer_question(question, top_k=5):
        return {
            "question": question,
            "answer": refusal,
            "refused": True,
            "sources": [
                {"source": "rag_survey.pdf", "page": 1, "score": 0.3}
            ],
        }

    monkeypatch.setattr(api_module, "answer_question", fake_answer_question)

    response = client.post(
        "/ask",
        json={"question": "Information not supported by the context"},
    )

    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["answer"] == refusal



