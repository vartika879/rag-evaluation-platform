from app.generation.llm import create_llm


def compress_context(query, results):
    llm = create_llm()

    context = "\n\n".join(
        result.payload.get("text", "")
        for result in results
    )

    prompt = f"""
Extract only the information from the context
that is directly relevant to the question.

Question:
{query}

Context:
{context}

Return only the relevant information.
"""

    response = llm.invoke(prompt)

    return response.content.strip()