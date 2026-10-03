from functools import lru_cache

from app.generation.llm import create_llm, generate_answer
from app.retrieval.retriever import (
    retrieve_chunks,
    build_context,
    build_sources,
)


@lru_cache(maxsize=1)
def get_llm():
    return create_llm()


def answer_question(question: str, top_k: int = 5) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    results = retrieve_chunks(question, top_k=top_k)
    sources = build_sources(results)

    if not results:
        return {
            "question": question,
            "answer": (
                "I don't have enough information in the provided documents."
            ),
            "refused": True,
            "sources": [],
        }

    context = build_context(results)
    result = generate_answer(get_llm(), question, context)

    return {
        "question": question,
        "answer": result["answer"],
        "refused": result["refused"],
        "sources": sources,
    }