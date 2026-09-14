from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

def rerank_results(query, results, top_k=5):
    
    if not results:
        return []

    pairs = []

    for result in results:
        text = result.get("text", "")

        if not isinstance(text, str):
            text = str(text)
        pairs.append([query, result["text"]])

    scores = model.predict(pairs, batch_size=4, show_progress_bar=False)
    reranked_results = []

    for result, score in zip(results, scores):
        result = result.copy()
        result["rerank_score"] = float(score)
        reranked_results.append(result)

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]
    
    