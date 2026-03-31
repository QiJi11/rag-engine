"""
RAG retrieval via simple vector store.

Interview point:
  "I abstract the vector store layer so we can swap implementations:
   MVP uses in-memory BM25 (no external deps), production scales to ChromaDB or Pinecone.
   Top-K=3 balances context window size with retrieval diversity."
"""

from store.vector_store import get_or_create_collection

COLLECTION_NAME = "knowledge_base"
TOP_K = 3


def add_documents(chunks: list[dict], source_name: str) -> None:
    """
    Add chunked documents to the vector store.

    Args:
        chunks: List of {text, chunk_index, source} dicts
        source_name: Identifier for the source document
    """
    collection = get_or_create_collection(COLLECTION_NAME)

    # Prepare data for insertion
    ids = [f"{source_name}_{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": source_name, "chunk_index": chunk["chunk_index"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


def retrieve(query: str, top_k: int = TOP_K) -> str:
    """
    Retrieve relevant context for a query.

    Args:
        query: User query string
        top_k: Number of chunks to retrieve

    Returns:
        Concatenated context string, or empty string if collection is empty
    """
    collection = get_or_create_collection(COLLECTION_NAME)

    count = collection.count()
    if count == 0:
        return ""

    # Retrieve top-k similar documents
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    # Concatenate retrieved texts
    context = "\n---\n".join(results["documents"][0])
    return context
