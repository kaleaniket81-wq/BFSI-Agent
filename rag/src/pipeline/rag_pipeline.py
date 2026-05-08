"""rag/src/pipeline/rag_pipeline.py — Retrieve relevant chunks, generate answer with LLM."""
import os
import time
import psycopg2
import psycopg2.extras
from typing import Dict, Optional
from dotenv import load_dotenv

from ingestion.src.storage.vector_store import VectorStore, get_connection
from llm.src.provider.base import LLMProviderFactory

load_dotenv()

BFSI_SYSTEM_PROMPT = """You are an expert BFSI (Banking, Financial Services and Insurance) analyst.
Answer questions using ONLY the provided document context.
- Be precise with numbers, dates, and financial figures
- Always cite which document and page you found the information on
- If the answer is not in the context, say "I don't have enough information in the provided documents."
- Never hallucinate financial data — accuracy is critical in BFSI
"""


class RAGPipeline:
    """Retrieve → Build context → Generate answer (fully local via Ollama)."""

    def __init__(self):
        self.store = VectorStore()
        self.llm   = LLMProviderFactory.get()

    def answer(self, question: str, doc_type: Optional[str] = None,
               top_k: int = 5) -> Dict:
        start = time.time()

        # Step 1: Embed the question
        q_embedding = self.llm.embed(question)

        # Step 2: Retrieve relevant chunks
        chunks = self.store.similarity_search(q_embedding, top_k=top_k, doc_type=doc_type)

        if not chunks:
            return {
                "answer":    "No relevant documents found. Please upload documents first.",
                "sources":   [],
                "provider":  self.llm.name,
                "latency_ms": 0,
            }

        # Step 3: Build context string
        context = ""
        sources = []
        for i, chunk in enumerate(chunks):
            context += (
                f"[Source {i+1}: {chunk['filename']}, page {chunk['page_number']},"
                f" type={chunk['doc_type']}, similarity={chunk['similarity']:.2f}]\n"
                f"{chunk['chunk_text']}\n\n"
            )
            src = f"{chunk['filename']} (page {chunk['page_number']})"
            if src not in sources:
                sources.append(src)

        # Step 4: Build prompt
        prompt = f"""Document context:
{context}

Question: {question}

Answer based ONLY on the context above. Cite sources as [Source N]."""

        # Step 5: Generate answer
        answer_text = self.llm.complete(prompt, system=BFSI_SYSTEM_PROMPT, temperature=0.05)

        latency_ms = int((time.time() - start) * 1000)

        # Step 6: Log to audit table
        self._log_query(question, answer_text, latency_ms)

        return {
            "answer":      answer_text,
            "sources":     sources,
            "chunks_used": len(chunks),
            "provider":    self.llm.name,
            "latency_ms":  latency_ms,
        }

    def _log_query(self, question: str, answer: str, latency_ms: int):
        try:
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO query_log (question, answer, llm_provider, latency_ms)
                       VALUES (%s, %s, %s, %s)""",
                    (question, answer, self.llm.name, latency_ms),
                )
        except Exception as e:
            print(f"[RAG] Audit log failed: {e}")
