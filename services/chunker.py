"""
Document text chunking with sliding window.
"""

import re


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks, preferring sentence/line boundaries.

    Args:
        text: Raw document text
        chunk_size: Target characters per chunk
        overlap: Overlap between consecutive chunks

    Returns:
        List of dicts: {text, chunk_index, source}
    """
    # Prefer sentence boundaries (。！？ or \n\n or newline)
    sentences = re.split(r'(。|！|？|\n\n|\n)', text)

    # Reconstruct sentences with delimiters
    reconstructed = []
    for i in range(0, len(sentences), 2):
        sent = sentences[i]
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
        reconstructed.append(sent + delimiter)

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in reconstructed:
        # If adding this sentence keeps us under chunk_size, add it
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            # Save current chunk and start a new one with overlap
            if current_chunk.strip():
                chunks.append(current_chunk)
                chunk_index += 1
                # Keep the last 'overlap' chars for continuity
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk += sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk)

    # Format as dicts
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "text": chunk_text.strip(),
            "chunk_index": i,
            "source": "uploaded_document",
        })

    return result
