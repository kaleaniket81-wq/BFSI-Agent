"""
rag/src/hybrid/hybrid_retriever.py

DAY 1 UPGRADE — Hybrid Retrieval
Combines:
  1. Dense vector search  (pgvector cosine similarity)
  2. Sparse keyword search (PostgreSQL full-text search / BM25)
  3. Merged via Reciprocal Rank Fusion (RRF)

Why hybrid beats pure vector:
  - Vector search misses exact matches: loan IDs, dates, rupee amounts
  - BM25 misses semantic matches: "penalty" vs "late payment charge"
  - RRF fuses both ranked lists without needing score normalisation

Interview explanation:
  "Pure vector search failed on queries like 'loan L-2024-001 EMI amount' because
   the vector space doesn't preserve exact token matches. BM25 catches those.
   I merged both lists using Reciprocal Rank Fusion — a parameter-free algorithm
   that's robust to score scale differences between retrieval methods."
"""

import os
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# RRF constant — 60 is standard from the original paper (Cormack et al. 2009)
RRF_K = 60


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "bfsi_intelligence"),
        user=os.getenv("DB_USER", "bfsi_user"),
        password=os.getenv("DB_PASSWORD"),
    )


class HybridRetriever:
    """
    Hybrid retrieval: dense (vector) + sparse (full-text) fused via RRF.
    """

    def __init__(self, top_k: int = 10, rrf_k: int = RRF_K):
        self.top_k = top_k
        self.rrf_k = rrf_k

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        doc_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run hybrid retrieval and return top_k fused results.

        Args:
            query:           Raw text query (used for full-text search)
            query_embedding: Dense embedding of query (used for vector search)
            doc_type:        Optional filter by document type
            top_k:           Override default top_k

        Returns:
            List of chunks sorted by RRF score (highest first)
        """
        k = top_k or self.top_k

        # Run both retrievers independently
        vector_results = self._vector_search(query_embedding, doc_type, k * 2)
        bm25_results   = self._fulltext_search(query, doc_type, k * 2)

        # Fuse and return top-k
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)
        return fused[:k]

    # ── Dense retrieval (pgvector) ────────────────────────────────────────────
    def _vector_search(
        self,
        embedding: List[float],
        doc_type: Optional[str],
        limit: int,
    ) -> List[Dict]:
        """Cosine similarity search via pgvector <=> operator."""
        type_filter = "AND d.doc_type = %s" if doc_type else ""
        params = [embedding, embedding, limit] if not doc_type else [embedding, doc_type, embedding, limit]

        sql = f"""
            SELECT
                dc.id::text                                          AS chunk_id,
                dc.chunk_text,
                dc.page_number,
                d.filename,
                d.doc_type,
                1 - (dc.embedding <=> %s::vector)                   AS score,
                'vector'                                             AS source
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.status = 'indexed'
              {type_filter}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """
        with get_connection() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # ── Sparse retrieval (PostgreSQL full-text / BM25 proxy) ─────────────────
    def _fulltext_search(
        self,
        query: str,
        doc_type: Optional[str],
        limit: int,
    ) -> List[Dict]:
        """
        PostgreSQL ts_rank_cd — a BM25-like ranking over tsvector index.
        Uses 'english' dictionary for stemming (loan→loan, penalties→penalty).
        """
        type_filter = "AND d.doc_type = %s" if doc_type else ""
        # Convert query to tsquery: handle multi-word queries
        ts_query = " & ".join(query.split())
        params   = [ts_query, ts_query, limit] if not doc_type else [ts_query, doc_type, ts_query, limit]

        sql = f"""
            SELECT
                dc.id::text                                          AS chunk_id,
                dc.chunk_text,
                dc.page_number,
                d.filename,
                d.doc_type,
                ts_rank_cd(
                    to_tsvector('english', dc.chunk_text),
                    to_tsquery('english', %s)
                )                                                    AS score,
                'bm25'                                               AS source
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.status = 'indexed'
              AND to_tsvector('english', dc.chunk_text)
                  @@ to_tsquery('english', %s)
              {type_filter}
            ORDER BY score DESC
            LIMIT %s
        """
        try:
            with get_connection() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        except Exception:
            # Invalid tsquery (e.g. special chars) — degrade gracefully
            return []

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
    ) -> List[Dict]:
        """
        RRF score = Σ 1 / (k + rank_in_list)

        A chunk ranked #1 in both lists scores highest.
        A chunk that only appears in one list still gets credit.

        Reference: Cormack, Clarke & Buettcher (SIGIR 2009)
        """
        scores: Dict[str, float] = {}
        chunks: Dict[str, Dict]  = {}

        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (self.rrf_k + rank)
            chunks[cid] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (self.rrf_k + rank)
            if cid not in chunks:
                chunks[cid] = chunk

        # Sort by fused RRF score
        sorted_ids = sorted(scores, key=lambda c: scores[c], reverse=True)
        result = []
        for cid in sorted_ids:
            chunk = chunks[cid].copy()
            chunk["rrf_score"]     = round(scores[cid], 6)
            chunk["retrieval_src"] = "hybrid"
            result.append(chunk)

        return result
