from typing import List

import chromadb
from chromadb.utils import embedding_functions
from schema import DocumentChunk

# 1. Setting up Persistent Storage
# This ensures my 3,190 chunks stay on disk even if i restart
CHROMA_DATA_PATH = (
    "d:/Users/ITORO SAMPSON/Documents/PYTHON PROJECTS/PRODUCTION_RAG_SYSTEM/chroma_db"
)
COLLECTION_NAME = "aws_well_architected"

# 2. Define the Embedding Function
# use SBERT (all-mpnet-base-v2) for high-quality semantic vectors
sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)


def get_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sbert_ef,
        metadata={"hnsw:space": "cosine"},  # Best for technical semantic search
    )


def add_chunks_to_db(chunks: List[DocumentChunk]):
    collection = get_vector_store()

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # CLEANING STEP
        sanitized_metadatas = []
        for c in batch:
            clean_meta = {
                k: (v if v is not None else "") for k, v in c.metadata.items()
            }
            sanitized_metadatas.append(clean_meta)

        collection.add(
            ids=[f"id_{c.metadata['chunk_index']}" for c in batch],
            documents=[c.content for c in batch],
            metadatas=sanitized_metadatas,  # Use the cleaned list here
        )
        print(f"[*] Indexed {i + len(batch)} / {len(chunks)} chunks...")

    print("[✓] Vector Store is ready and persistent on D: drive.")


if __name__ == "__main__":
    # This is for testing the connection
    store = get_vector_store()
    print(f"Connected to collection. Current count: {store.count()}")
