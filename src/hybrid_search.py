from .keyword_search import keyword_search
from .dense_search import dense_search


def hybrid_search(query, limit=20):
    bm25_results = keyword_search(query, limit)
    dense_results = dense_search(query, limit)

    k = 60
    rrf_scores = {}
    chunk_data = {}

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = (result["source"], result["chunk_id"])
        score = 1 / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score
        chunk_data[chunk_id] = result

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = (result["source"], result["chunk_id"])
        score = 1 / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score
        chunk_data[chunk_id] = result

  
    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []
    for chunk_id, score in ranked_chunks[:limit]:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = score
        results.append(result)

    return results

      