from app.generation.llm import create_llm

def generate_queries(query:str,num_queries:int=3):

    llm=create_llm()

    prompt = f"""
Generate {num_queries} different search queries for the same information need.

Original question:
{query}

Return only the queries, one per line.
"""
    response = llm.invoke(prompt)

    return [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()

    ]