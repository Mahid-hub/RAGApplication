import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from .ingest import load_documents
from .chunker import recursive_chunk_text
from .embeddings import embed_text

load_dotenv()

def get_client():
    endpoint = os.getenv("Endpoint")
    api_key = os.getenv("cluster-api-key")

    return QdrantClient(
        url=endpoint,
        api_key=api_key,
        timeout=60
    )


def index_documents():
    client = get_client()
    collection_name = os.getenv("collection-name")
    chunk_size = int(os.getenv("chunk-size"))
    documents = load_documents("data/raw")

    if client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists.")
        print("Deleting old collection for clean re-indexing...")
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' created.")

    point_number = 1
    for doc in documents:
        chunks = recursive_chunk_text(doc["text"], chunk_size, doc)
        print(f"\nIndexing: {doc['metadata']['source']}")
        print(f"Chunks: {len(chunks)}")

        for chunk in chunks:
            chunk_id = str(point_number)            # Use global point number as chunk ID
            embedding = embed_text(chunk["text"])
            point = PointStruct(
                id=point_number,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "document_title": chunk["document_title"],
                    "source": chunk["source"],
                    "text": chunk["text"]
                }
            )
            client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            point_number += 1

    print("\n" + "=" * 60)
    print("DOCUMENT INDEXING COMPLETED")
    print("=" * 60)
    print(f"Documents indexed: {len(documents)}")
    print(f"Total points: {point_number - 1}")
