from app.generation.llm import create_llm
from app.retrieval.retriever import retrieve_chunks

def evaluate_context(query, results):
    llm = create_llm()

    context = "\n\n".join(
        result.payload.get("text", "")
        for result in results
    )

    prompt = f"""
Evaluate whether the retrieved context is useful for answering the question.

Question:
{query}

Context:
{context}

Return only:
GOOD
or
BAD
"""

    response = llm.invoke(prompt)

    return response.content.strip().upper()



def corrective_retrieval(query: str, top_k: int = 5):
    return retrieve_chunks(
        query,
        top_k=top_k * 2,
    )



def has_supporting_evidence(results, threshold=0.30):
    if not results:
        return False

    return any(
        result.score >= threshold
        for result in results
    )