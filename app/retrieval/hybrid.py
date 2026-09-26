def min_max_normalize(scores):
    if not scores:
        return []

    min_score=min(scores)

    max_score=max(scores)

    if max_score == min_score:
        return[1.0]*len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def hybrid_search(vector_results,
    bm25_results,
    top_k=5,
    vector_weight=0.5):
    bm25_weight= 1-vector_weight

     # Normalize vector scores
    vector_score_values = [
        float(result.score)
        for result in vector_results
    ]

    normalized_vector_scores = min_max_normalize(
        vector_score_values
    )

    vector_map = {
        result.payload["chunk_id"]: normalized_score
        for result, normalized_score in zip(
            vector_results,
            normalized_vector_scores,
        )
    }

    # Normalize BM25 scores
    bm25_score_values = [
        result["score"]
        for result in bm25_results
    ]

    normalized_bm25_scores = min_max_normalize(
        bm25_score_values
    )

    bm25_map = {
        result["chunk_id"]: normalized_score
        for result, normalized_score in zip(
            bm25_results,
            normalized_bm25_scores,
        )
    }
     # Combine both retrieval signals
    all_chunk_ids = set(vector_map) | set(bm25_map)

    ranked_results = []

    for chunk_id in all_chunk_ids:
        vector_score = vector_map.get(chunk_id, 0.0)
        bm25_score = bm25_map.get(chunk_id, 0.0)

        hybrid_score = (
            vector_weight * vector_score
            + bm25_weight * bm25_score
        )
        ranked_results.append(
            {
                "chunk_id": chunk_id,
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score,
            }
        )

    ranked_results.sort(
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )

    return ranked_results[:top_k]
