from app.generation.llm import create_llm

def transform_query(query:str, context:str)->str:
    llm=create_llm()

    prompt=  f"""
Rewrite the question into a clear, standalone search query.

Context:
{context}

Question:
{query}

Return only the rewritten query.
"""

    response=llm.invoke(prompt)

    return response.content.strip()
