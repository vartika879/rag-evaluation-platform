from rank_bm25 import BM25Okapi

def create_bm25_index(chunks):
    tokenized_chunks=[
        chunk.page_content.lower().split()
        for chunk in chunks 
    ]

    return BM25Okapi(tokenized_chunks)

def retrieve_bm25(
    bm25,
    chunks,
    query: str,
    top_k: int = 5,
):

    tokenized_query = query.lower().split()

    scores=bm25.get_scores(tokenized_query)

    ranked_indices=sorted(
        range(len(scores)),
        key=lambda i:scores[i],
        reverse=True,

    )[:top_k]

    results = []

    for index in ranked_indices:
        chunk=chunks[index]


        results.append(

            {
                "chunk": chunk,
                "chunk_id": chunk.metadata["chunk_id"],
                "score": float(scores[index]),
            }
        )

    return results