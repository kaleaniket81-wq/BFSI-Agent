"""ingestion/src/pipeline.py — Orchestrate parse → chunk → embed → store."""
import os
import time
from typing import Dict
from dotenv import load_dotenv

from ingestion.src.parser.document_parser import DocumentParser
from ingestion.src.chunker.text_chunker import TextChunker
from ingestion.src.storage.vector_store import VectorStore
from llm.src.provider.base import LLMProviderFactory

load_dotenv()


class IngestionPipeline:
    """End-to-end document ingestion for BFSI files."""

    def __init__(self):
        self.parser  = DocumentParser()
        self.chunker = TextChunker(chunk_size=400, overlap=80)
        self.store   = VectorStore()
        self.llm     = LLMProviderFactory.get()

    def ingest(self, filepath: str, doc_type: str = "general") -> Dict:
        """
        Full pipeline:
        1. Parse PDF/DOCX to text
        2. Chunk text with overlap
        3. Embed each chunk via Ollama (or AWS Bedrock)
        4. Store in PostgreSQL + pgvector
        """
        start = time.time()
        print(f"[Ingest] Starting: {filepath} | type={doc_type} | provider={self.llm.name}")

        # Step 1: Parse
        parsed = self.parser.parse(filepath)
        parsed["doc_type"] = doc_type
        print(f"[Ingest] Parsed {parsed['page_count']} pages from {parsed['filename']}")

        # Step 2: Save document record
        file_size = os.path.getsize(filepath)
        doc_id = self.store.save_document(
            filename=parsed["filename"],
            doc_type=doc_type,
            page_count=parsed["page_count"],
            file_size=file_size,
        )

        # Step 3: Chunk
        chunks = self.chunker.chunk_document(parsed)
        print(f"[Ingest] Created {len(chunks)} chunks")

        # Step 4: Embed all chunks
        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = self.llm.embed(chunk["text"])
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"[Ingest] Embedded {i+1}/{len(chunks)} chunks")

        # Step 5: Store chunks + embeddings
        saved = self.store.save_chunks(doc_id, chunks, embeddings)

        elapsed = round(time.time() - start, 2)
        print(f"[Ingest] Done in {elapsed}s — saved {saved} chunks for doc_id={doc_id}")

        return {
            "document_id": doc_id,
            "filename":    parsed["filename"],
            "doc_type":    doc_type,
            "pages":       parsed["page_count"],
            "chunks":      saved,
            "llm_provider": self.llm.name,
            "elapsed_sec": elapsed,
        }
