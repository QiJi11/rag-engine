"""
RAG retrieval via ChromaDB and Reranking.

Interview point:
  "I fetch Top-20 candidate chunks from the vector database (fast but coarse),
   then use a CrossEncoder to perform semantic Reranking (slow but highly accurate) 
   to select the final Top-3 chunks. This dramatically improves retrieval quality 
   while balancing compute cost.
   
   I also use lazy loading for the reranker model — it only loads on first use,
   so the server starts instantly even without the model downloaded."
"""

import os
from store.vector_store import get_or_create_collection

COLLECTION_NAME = "knowledge_base"
TOP_K = 3
FETCH_K = 15  # For coarse retrieval

# Lazy-loaded reranker (avoids blocking server startup if model not cached)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # Use mirror if HF not accessible
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _reranker


def add_documents(chunks: list[dict], source_name: str) -> None:
    """Add chunked documents to the vector store."""
    collection = get_or_create_collection(COLLECTION_NAME)
    ids = [f"{source_name}_{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": source_name, "chunk_index": chunk["chunk_index"]} for chunk in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def retrieve(query: str, top_k: int = TOP_K) -> str:
    """
    Retrieve relevant context using Two-Stage Retrieval:
    1. Vector Search (Coarse) -> Top 15
    2. Rerank (Fine) -> Top 3
    """
    collection = get_or_create_collection(COLLECTION_NAME)
    count = collection.count()
    if count == 0:
        return ""

    # Phase 1: Vector Retrieval (Coarse)
    results = collection.query(query_texts=[query], n_results=min(FETCH_K, count))
    docs = results["documents"][0] if results["documents"] else []
    if not docs:
        return ""

    if len(docs) <= top_k:
        return "\n---\n".join(docs)

    # Phase 2: Reranking (Fine) — loaded lazily
    reranker = get_reranker()
    rerank_pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(rerank_pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:top_k]]
    return "\n---\n".join(top_docs)