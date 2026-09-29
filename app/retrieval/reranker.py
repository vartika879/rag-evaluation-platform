import os 
import cohere 
from dotenv import load_dotenv

load_dotenv()

def rerank_results(query,results,top_k=3):

    client=cohere.Client(os.getenv("COHERE_API_KEY"))

    documents=[
        result.payload.get("text", "")
        for result in results
    ]

    response = client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_k,
    )

    reranked =[]

    for item in response.results:
        result=results[item.index]

        reranked.append({
            "chunk_id": result.payload["chunk_id"],
            "page": result.payload.get("page"),
            "score": item.relevance_score,
            "text": result.payload.get("text", ""),
        })

    return reranked