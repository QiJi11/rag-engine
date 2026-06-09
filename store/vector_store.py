"""
ChromaDB vector store implementation.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# Store database on disk
STORE_DIR = "./data/chroma_db"

# Module-level cache (loaded once, reused)
_collection_cache: dict = {}

def get_or_create_collection(name: str):
    """
    Get or create a ChromaDB collection using a local embedding model.
    Uses lazy loading to avoid blocking server startup.
    """
    if name in _collection_cache:
        return _collection_cache[name]

    # Use HF mirror if needed
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    client = chromadb.PersistentClient(path=STORE_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(name=name, embedding_function=emb_fn)
    _collection_cache[name] = collection
    return collection
