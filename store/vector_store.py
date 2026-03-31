"""
Simple in-memory vector store (no external deps) for MVP.
In production, would use ChromaDB or Pinecone.

Interview point:
  "For MVP, I use in-memory storage to avoid dependency hell.
   Production would swap in ChromaDB (persistent) or Pinecone (cloud).
   The interface is abstracted so switching is painless."
"""

import json
import os
from pathlib import Path

# Store metadata on disk for persistence across restarts
STORE_FILE = "./data/vector_store.json"


def _ensure_dir():
    """Create data directory if needed."""
    Path("./data").mkdir(exist_ok=True)


def _load_store() -> dict:
    """Load store from disk, or empty dict if none."""
    _ensure_dir()
    if os.path.exists(STORE_FILE):
        try:
            with open(STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_store(store: dict):
    """Persist store to disk."""
    _ensure_dir()
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def get_or_create_collection(name: str):
    """
    Get or create a collection (returns simple dict interface).
    Mimics ChromaDB collection API for compatibility.
    """
    store = _load_store()
    if name not in store:
        store[name] = {"documents": {}, "metadatas": {}}
    return VectorCollection(name, store)


class VectorCollection:
    """Simple in-memory collection with persistence."""

    def __init__(self, name: str, store: dict):
        self.name = name
        self.store = store

    def add(self, ids: list, documents: list, metadatas: list):
        """Add documents to collection."""
        col = self.store[self.name]
        for doc_id, text, meta in zip(ids, documents, metadatas):
            col["documents"][doc_id] = text
            col["metadatas"][doc_id] = meta
        _save_store(self.store)

    def query(self, query_texts: list, n_results: int):
        """Simple BM25-like keyword matching (no embeddings)."""
        col = self.store[self.name]
        query = query_texts[0].lower()
        query_words = set(query.split())

        # Score each document
        scores = {}
        for doc_id, text in col["documents"].items():
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scores[doc_id] = overlap

        # Return top-n
        top_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        top_ids = [id for id, _ in top_ids]

        documents = [[col["documents"][id] for id in top_ids]]
        return {"documents": documents}

    def count(self) -> int:
        """Return document count."""
        return len(self.store[self.name]["documents"])

    def get(self):
        """Get all documents (for listing sources)."""
        col = self.store[self.name]
        ids = list(col["documents"].keys())
        documents = list(col["documents"].values())
        metadatas = list(col["metadatas"].values())
        return {"ids": ids, "documents": documents, "metadatas": metadatas}
