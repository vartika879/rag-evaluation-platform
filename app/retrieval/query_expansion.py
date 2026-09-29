from app.generation.llm import create_llm


def expand_query(query: str):
    llm = create_llm()

    prompt = f"""
Expand this search query with useful related terms.

Original query:
{query}

Return only one expanded search query.
"""

    response = llm.invoke(prompt)

    return response.content.strip()