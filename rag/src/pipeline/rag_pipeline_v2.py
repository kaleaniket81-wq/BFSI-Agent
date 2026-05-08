"""
rag/src/pipeline/rag_pipeline_v2.py

DAY 1 UPGRADE — Full upgraded RAG pipeline:
  query → rewrite → hybrid retrieve (vector + BM25 + RRF) → generate

Replaces rag_pipeline.py for all new queries.
Old pipeline kept for backward compatibility.
"""

import os
import time
import psycopg2
from typing import Dict, List, Optional
from dotenv import load_dotenv

from llm.src.provider.base import LLMProviderFactory
from rag.src.hybrid.hybrid_retriever import HybridRetriever
from rag.src.rewriter.query_rewriter import QueryRewriter
from ingestion.src.storage.vector_store import get_connection

load_dotenv()

BFSI_SYSTEM_PROMPT = """You are an expert BFSI (Banking, Financial Services and Insurance) analyst.
Answer questions using ONLY the provided document context.
- Be precise with numbers, dates, and financial figures
- Always cite which document and page the information came from [Source N]
- If the answer is not in the context, say "I don't have enough information in the provided documents"
- Never hallucinate financial data — accuracy is critical in BFSI
- Format amounts in Indian currency style (₹X,XX,XXX)
"""


class RAGPipelineV2:
    """
    Production RAG pipeline with:
    - Query rewriting (expand vague queries)
    - Hybrid retrieval (vector + BM25 + RRF)
    - Latency breakdown per stage
    - Full audit logging
    """

    def __init__(self):
        self.llm       = LLMProviderFactory.get()
        self.retriever = HybridRetriever(top_k=10)
        self.rewriter  = QueryRewriter(llm_provider=self.llm)

    def answer(
        self,
        question: str,
        doc_type:     Optional[str] = None,
        top_k:        int           = 5,
        chat_history: List[dict]    = None,
    ) -> Dict:
        timings = {}
        total_start = time.time()

        # ── Stage 1: Query rewriting ─────────────────────────────────────────
        t = time.time()
        if chat_history:
            rewritten_query = self.rewriter.rewrite_with_context(question, chat_history)
        else:
            rewritten_query = self.rewriter.rewrite(question)
        timings["rewrite_ms"] = int((time.time() - t) * 1000)

        # ── Stage 2: Embed rewritten query ───────────────────────────────────
        t = time.time()
        query_embedding = self.llm.embed(rewritten_query)
        timings["embed_ms"] = int((time.time() - t) * 1000)

        # ── Stage 3: Hybrid retrieval ────────────────────────────────────────
        t = time.time()
        chunks = self.retriever.retrieve(
            query           = rewritten_query,
            query_embedding = query_embedding,
            doc_type        = doc_type,
            top_k           = top_k,
        )
        timings["retrieval_ms"] = int((time.time() - t) * 1000)

        if not chunks:
            return {
                "answer":          "No relevant documents found. Please upload BFSI documents first.",
                "sources":         [],
                "original_query":  question,
                "rewritten_query": rewritten_query,
                "provider":        self.llm.name,
                "timings":         timings,
                "latency_ms":      int((time.time() - total_start) * 1000),
            }

        # ── Stage 4: Build context ───────────────────────────────────────────
        context = ""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            context += (
                f"[Source {i}: {chunk['filename']}, "
                f"page {chunk['page_number']}, "
                f"type={chunk['doc_type']}, "
                f"rrf_score={chunk.get('rrf_score', 0):.4f}]\n"
                f"{chunk['chunk_text']}\n\n"
            )
            src = f"{chunk['filename']} (page {chunk['page_number']})"
            if src not in sources:
                sources.append(src)

        prompt = f"""Document context:
{context}

Original question: {question}
Refined question:  {rewritten_query}

Answer based ONLY on the context. Cite sources as [Source N]."""

        # ── Stage 5: Generate answer ─────────────────────────────────────────
        t = time.time()
        answer_text = self.llm.complete(prompt, system=BFSI_SYSTEM_PROMPT, temperature=0.05)
        timings["llm_ms"] = int((time.time() - t) * 1000)

        total_ms = int((time.time() - total_start) * 1000)
        timings["total_ms"] = total_ms

        # ── Stage 6: Audit log ───────────────────────────────────────────────
        self._log(question, rewritten_query, answer_text, total_ms)

        return {
            "answer":          answer_text,
            "sources":         sources,
            "chunks_used":     len(chunks),
            "original_query":  question,
            "rewritten_query": rewritten_query,
            "provider":        self.llm.name,
            "timings":         timings,
            "latency_ms":      total_ms,
        }

    def _log(self, question: str, rewritten: str, answer: str, latency_ms: int):
        try:
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO query_log
                       (question, answer, llm_provider, latency_ms)
                       VALUES (%s, %s, %s, %s)""",
                    (f"[rewritten: {rewritten}] {question}", answer,
                     self.llm.name, latency_ms),
                )
        except Exception as e:
            print(f"[RAGv2] Audit log failed: {e}")
