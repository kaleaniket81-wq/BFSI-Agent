"""ingestion/src/chunker/text_chunker.py — Sliding-window chunker with overlap."""
from typing import List, Dict


class TextChunker:
    """Split text into overlapping chunks for optimal vector retrieval."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        # chunk_size: max words per chunk (400 ≈ ~300 tokens — fits nomic-embed-text)
        # overlap:    shared words between adjacent chunks to preserve context
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk_page(self, text: str, page_number: int, doc_type: str = "") -> List[Dict]:
        """Chunk a single page's text."""
        words = text.split()
        if not words:
            return []

        chunks, start = [], 0
        while start < len(words):
            end        = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "text":        chunk_text,
                "page_number": page_number,
                "chunk_index": len(chunks),
                "token_count": len(words[start:end]),
                "doc_type":    doc_type,
            })
            if end == len(words):
                break
            start = end - self.overlap
        return chunks

    def chunk_document(self, parsed_doc: Dict) -> List[Dict]:
        """Chunk all pages of a parsed document."""
        doc_type = parsed_doc.get("doc_type", "")
        all_chunks = []
        for page in parsed_doc.get("pages", []):
            page_chunks = self.chunk_page(
                page["text"], page["page_number"], doc_type
            )
            all_chunks.extend(page_chunks)
        return all_chunks
