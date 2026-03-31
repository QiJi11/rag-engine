"""
File upload and document ingestion router.

Interview point:
  "Multi-format support (TXT, PDF) with proper error handling.
   Once documents are chunked and indexed, the RAG retriever can serve
   context for any query without re-processing."
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pypdf import PdfReader
from services.chunker import chunk_text
from services.rag import add_documents
from store.vector_store import get_or_create_collection

router = APIRouter()

COLLECTION_NAME = "knowledge_base"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document (TXT or PDF).

    Returns: {filename, chunks_count}
    """
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    content = await file.read()

    # Extract text based on file type
    if file.filename.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "TXT file must be UTF-8 encoded")
    elif file.filename.endswith(".pdf"):
        try:
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(content))
            text = "".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse PDF: {str(e)}")
    else:
        raise HTTPException(400, "File must be .txt or .pdf")

    if not text.strip():
        raise HTTPException(400, "File is empty or contains no extractable text")

    # Chunk and index
    chunks = chunk_text(text)
    add_documents(chunks, file.filename)

    return {
        "filename": file.filename,
        "chunks_count": len(chunks),
    }


@router.get("/docs")
async def list_documents():
    """
    List all document sources in the knowledge base.

    Returns: {sources: [str]}
    """
    collection = get_or_create_collection(COLLECTION_NAME)

    # Get all documents and extract unique sources
    results = collection.get()
    sources = set()

    if results["metadatas"]:
        for metadata in results["metadatas"]:
            if "source" in metadata:
                sources.add(metadata["source"])

    return {"sources": sorted(list(sources))}
