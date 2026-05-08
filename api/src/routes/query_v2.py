"""
api/src/routes/query_v2.py — Upgraded query route using all Day 1–5 improvements.

Replaces query.py for production use.
Wires together: rewrite → hybrid retrieve → cache → generate → monitor
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2", tags=["Query v2 — Upgraded"])


class QueryV2Request(BaseModel):
    question:  str           = Field(..., min_length=3)
    doc_type:  Optional[str] = None
    top_k:     int           = Field(5, ge=1, le=20)
    use_cache: bool          = True


class ExtractRequest(BaseModel):
    document_text: str = Field(..., min_length=50)


class CompareRequest(BaseModel):
    doc1_text: str = Field(..., min_length=50)
    doc2_text: str = Field(..., min_length=50)
    doc1_name: str = "Document 1"
    doc2_name: str = "Document 2"


# ── Lazy singletons (initialised once on first request) ───────────────────────
_pipeline   = None
_extractor  = None
_cache      = None
_metrics    = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag.src.pipeline.rag_pipeline_v2 import RAGPipelineV2
        from cache.src.redis_cache import RedisCache, CachedLLMProvider
        from llm.src.provider.base import LLMProviderFactory

        cache    = _get_cache()
        base_llm = LLMProviderFactory.get()
        cached_llm = CachedLLMProvider(base_llm, cache)

        _pipeline = RAGPipelineV2()
        _pipeline.llm = cached_llm   # inject cached provider
    return _pipeline


def _get_extractor():
    global _extractor
    if _extractor is None:
        from extraction.src.bfsi_extractor import BFSIExtractor
        _extractor = BFSIExtractor()
    return _extractor


def _get_cache():
    global _cache
    if _cache is None:
        from cache.src.redis_cache import RedisCache
        _cache = RedisCache()
    return _cache


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query")
def query_v2(request: QueryV2Request):
    """
    Upgraded RAG query:
    1. Validates input
    2. Checks Redis query cache
    3. Rewrites query for better retrieval
    4. Hybrid search (vector + BM25 + RRF)
    5. Generates grounded answer
    6. Caches result
    7. Logs latency metrics
    """
    from monitoring.src.metrics import failures, metrics

    # Input validation
    err = failures.validate_query(request.question)
    if err:
        raise HTTPException(400, err)

    cache = _get_cache()

    # Cache lookup
    if request.use_cache:
        cached = cache.get_query_result(request.question, request.doc_type)
        if cached:
            cached["cache_hit"] = True
            return cached

    # Run pipeline
    try:
        pipeline = _get_pipeline()
        result   = pipeline.answer(
            question = request.question,
            doc_type = request.doc_type,
            top_k    = request.top_k,
        )
    except TimeoutError:
        raise HTTPException(504, "LLM timeout — try again or switch to bedrock provider")
    except Exception as e:
        metrics.record_error("rag_pipeline", str(e))
        raise HTTPException(500, f"Query failed: {str(e)}")

    # Handle empty retrieval
    if not result.get("sources"):
        return failures.handle_empty_retrieval(request.question, request.doc_type)

    result["cache_hit"] = False

    # Cache result
    if request.use_cache:
        cache.set_query_result(request.question, request.doc_type, result)

    # Log metrics
    metrics.record_query(
        question    = request.question,
        timings     = result.get("timings", {}),
        provider    = result.get("provider", "unknown"),
        chunks_used = result.get("chunks_used", 0),
        cache_hit   = False,
    )

    return result


@router.post("/extract")
def extract_fields(request: ExtractRequest):
    """
    Extract structured BFSI fields from raw document text.
    Returns: loan_id, interest_rate, tenure, emi, penalties, collateral etc.
    """
    try:
        extractor = _get_extractor()
        extracted = extractor.extract_loan_fields(request.document_text)
        compliance = extractor.check_rbi_compliance(extracted)
        return {"extracted": extracted, "compliance": compliance}
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {str(e)}")


@router.post("/compare")
def compare_documents(request: CompareRequest):
    """
    Compare two BFSI documents side-by-side.
    Returns: field-by-field comparison + recommendation.
    """
    try:
        extractor = _get_extractor()
        return extractor.compare_documents(
            request.doc1_text, request.doc2_text,
            request.doc1_name, request.doc2_name,
        )
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {str(e)}")


@router.get("/cache/stats")
def cache_stats():
    """Redis cache statistics — hit rate, hits, misses."""
    return _get_cache().stats()


@router.delete("/cache/invalidate")
def invalidate_cache():
    """Invalidate all query caches — call after bulk document ingestion."""
    _get_cache().invalidate_query_cache()
    return {"message": "Query cache invalidated"}
