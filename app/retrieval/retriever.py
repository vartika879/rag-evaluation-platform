from app.embeddings.service import create_embedding_model
from app.vectorstore.qdrant_store import COLLECTION_NAME,create_qdrant_client

def retrieve_chunks(query:str,top_k:int=5):
    embedding_model=create_embedding_model()
    query_vector=embedding_model.embed_query(query)

    client=create_qdrant_client()

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return results.points

    finally:

        client.close()


def build_context(results):
    context_parts=[]

    for i,result in enumerate(results,start=1):
        text=result.payload.get("text","")
        source = result.payload.get("source", "")
        page = result.payload.get("page")

        context_parts.append(
            f"[Source {i}]\n"
            f"File: {source}\n"
            f"Page: {page + 1 if page is not None else 'Unknown'}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)

def build_sources(results):
    sources=[]

    seen=set()

    for result in results:

        source= result.payload.get("source","")
        page=result.payload.get("page")
        page_number = page + 1 if page is not None else None

        key = (source, page_number)

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page_number,
                "score": result.score,
            }
        )

    return sources