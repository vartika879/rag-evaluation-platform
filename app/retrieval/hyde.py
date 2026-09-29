from app.generation.llm import create_llm
from app.embeddings.service import create_embedding_model


def generate_hypothetical_document(query: str):
    llm = create_llm()

    prompt = f"""
Write a short hypothetical answer to this question.

Question:
{query}

Return only the answer.
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def create_hyde_embedding(query: str):
    hypothetical_document = generate_hypothetical_document(query)

    embedding_model = create_embedding_model()
    embedding = embedding_model.embed_query(
        hypothetical_document
    )

    return hypothetical_document, embedding