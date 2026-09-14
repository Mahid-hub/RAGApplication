from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_text(chunk):
    embedding = model.encode(chunk)
    return embedding
    
       