import os
from .embeddings import embed_text
from .qdrant_db import get_client, index_documents

client = get_client()

def dense_search(user_query, limit):
    collection_name = os.getenv('collection-name')
    if not collection_name:
        raise ValueError("Collection name is not set in the environment variables.")

    if not client.collection_exists(collection_name=collection_name):
        # raise ValueError("Collection name is not exist.")
        index_documents()
    
    embed_query = embed_text(user_query)
    result = client.query_points(
        collection_name=collection_name,
        query=embed_query,
        limit=limit,
        with_payload=True,
        timeout=60
    )
    
    results = []
    for point in result.points:
        result_item = {
            "chunk_id": point.payload["chunk_id"],
            "document_title": point.payload["document_title"],
            "source": point.payload["source"],
            "text": point.payload["text"],
            "score": point.score
        }

        results.append(result_item)

    return results
