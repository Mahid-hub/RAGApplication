import os
from dotenv import load_dotenv
from .ingest import load_documents
from .chunker import recursive_chunk_text
from rank_bm25 import BM25Okapi

load_dotenv()

size = int(os.getenv("chunk-size"))
documents = load_documents("data/raw")

all_chunks = []
chunk_number = 1

for doc in documents:
    chunks = recursive_chunk_text(doc["text"], size, doc)
    # all_chunks.extend(chunks)
    for chunk in chunks:
        chunk["chunk_id"] = str(chunk_number)
        all_chunks.append(chunk)
        chunk_number += 1

tokenized_chunks = []
for chunk in all_chunks:
    text = chunk["text"]
    tokenized_chunks.append(text.lower().split())
    
bm25 = BM25Okapi(tokenized_chunks)

def keyword_search(query, limit):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i:scores[i],
        reverse=True
    )

    results = []
    for index in ranked_indexes[:limit]:
        result = {
            "chunk_id": all_chunks[index]["chunk_id"],
            "document_title": all_chunks[index]["document_title"],
            "source": all_chunks[index]["source"],
            "text": all_chunks[index]["text"],
            "score": float(scores[index])
        }

        results.append(result)

    return results

