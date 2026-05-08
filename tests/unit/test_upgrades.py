"""tests/unit/test_upgrades.py — Unit tests for Day 1-5 upgrades."""
import math
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Day 1: Smart Chunker
# ─────────────────────────────────────────────────────────────────────────────
class TestSmartChunker:

    def test_fixed_strategy(self):
        from ingestion.src.chunker.smart_chunker import SmartChunker
        chunker = SmartChunker(strategy="fixed")
        parsed  = {
            "doc_type": "loan_agreement",
            "pages": [{"page_number": 1, "text": " ".join([f"word{i}" for i in range(50)])}]
        }
        chunks = chunker.chunk_document(parsed)
        assert len(chunks) > 0
        assert all(c["strategy"] == "fixed" for c in chunks)

    def test_paragraph_strategy_preserves_clauses(self):
        from ingestion.src.chunker.smart_chunker import SmartChunker
        chunker = SmartChunker(strategy="paragraph")
        text    = "Clause 1: The borrower agrees to pay EMI.\n\nClause 2: Interest rate is 8.5%."
        parsed  = {"doc_type": "loan_agreement", "pages": [{"page_number": 1, "text": text}]}
        chunks  = chunker.chunk_document(parsed)
        assert len(chunks) >= 1
        assert all(c["strategy"] == "paragraph" for c in chunks)

    def test_chunk_index_is_global(self):
        from ingestion.src.chunker.smart_chunker import SmartChunker
        chunker = SmartChunker(strategy="fixed")
        parsed  = {
            "doc_type": "general",
            "pages": [
                {"page_number": 1, "text": " ".join([f"a{i}" for i in range(30)])},
                {"page_number": 2, "text": " ".join([f"b{i}" for i in range(30)])},
            ]
        }
        chunks = chunker.chunk_document(parsed)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks))), "chunk_index must be sequential globally"

    def test_empty_page_skipped(self):
        from ingestion.src.chunker.smart_chunker import SmartChunker, FixedChunker
        chunker = FixedChunker()
        assert chunker.chunk("", page_number=1) == []
        assert chunker.chunk("   ", page_number=1) == []


# ─────────────────────────────────────────────────────────────────────────────
# Day 1: Hybrid Retriever — RRF
# ─────────────────────────────────────────────────────────────────────────────
class TestHybridRetriever:

    def _make_chunk(self, id_, text, score=0.8):
        return {"chunk_id": id_, "chunk_text": text, "page_number": 1,
                "filename": "test.pdf", "doc_type": "loan_agreement", "score": score}

    def test_rrf_higher_score_for_dual_ranked(self):
        from rag.src.hybrid.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()

        # chunk_A appears #1 in both lists — should score highest
        chunk_A = self._make_chunk("A", "EMI is 10234")
        chunk_B = self._make_chunk("B", "Interest rate 8.5")
        chunk_C = self._make_chunk("C", "Penalty clause")

        vector_results = [chunk_A, chunk_B, chunk_C]
        bm25_results   = [chunk_A, chunk_C, chunk_B]

        fused = retriever._reciprocal_rank_fusion(vector_results, bm25_results)
        assert fused[0]["chunk_id"] == "A", "Dual top-ranked chunk should win"
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    def test_rrf_includes_bm25_only_results(self):
        from rag.src.hybrid.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()

        chunk_A = self._make_chunk("A", "vector only")
        chunk_B = self._make_chunk("B", "bm25 only")

        fused = retriever._reciprocal_rank_fusion([chunk_A], [chunk_B])
        ids   = [c["chunk_id"] for c in fused]
        assert "A" in ids and "B" in ids, "Both sources must appear in fused result"

    def test_rrf_score_formula(self):
        from rag.src.hybrid.hybrid_retriever import HybridRetriever, RRF_K
        retriever = HybridRetriever()

        chunk = self._make_chunk("X", "test")
        # rank 1 in vector, rank 1 in bm25
        fused = retriever._reciprocal_rank_fusion([chunk], [chunk])
        expected_score = 2 / (RRF_K + 1)   # 1/(k+1) + 1/(k+1)
        assert abs(fused[0]["rrf_score"] - expected_score) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Day 1: Query Rewriter
# ─────────────────────────────────────────────────────────────────────────────
class TestQueryRewriter:

    def test_rewrites_short_vague_query(self):
        from rag.src.rewriter.query_rewriter import QueryRewriter
        mock_provider = MagicMock()
        mock_provider.complete.return_value = (
            "What are the late payment penalties specified in the loan agreement?"
        )
        rewriter  = QueryRewriter(llm_provider=mock_provider)
        result    = rewriter.rewrite("penalty?")
        assert len(result) > len("penalty?")
        mock_provider.complete.assert_called_once()

    def test_skips_rewrite_for_long_query(self):
        from rag.src.rewriter.query_rewriter import QueryRewriter
        mock_provider = MagicMock()
        rewriter  = QueryRewriter(llm_provider=mock_provider)
        long_q    = "What is the total EMI amount due every month " * 3  # > 20 words
        result    = rewriter.rewrite(long_q)
        assert result == long_q   # unchanged
        mock_provider.complete.assert_not_called()

    def test_returns_original_on_llm_failure(self):
        from rag.src.rewriter.query_rewriter import QueryRewriter
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = Exception("LLM timeout")
        rewriter = QueryRewriter(llm_provider=mock_provider)
        result   = rewriter.rewrite("EMI amount")
        assert result == "EMI amount"

    def test_no_provider_returns_original(self):
        from rag.src.rewriter.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(llm_provider=None)
        assert rewriter.rewrite("penalty") == "penalty"


# ─────────────────────────────────────────────────────────────────────────────
# Day 2: Evaluator metrics
# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluatorMetrics:

    def _cosine(self, a, b):
        dot  = sum(x * y for x, y in zip(a, b))
        na   = math.sqrt(sum(x**2 for x in a))
        nb   = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb) if na * nb else 0.0

    def test_token_overlap_identical(self):
        from evaluation.src.evaluator import RAGEvaluator
        ev = RAGEvaluator.__new__(RAGEvaluator)
        score = ev._token_overlap("EMI is ten thousand", "EMI is ten thousand")
        assert score == 1.0

    def test_token_overlap_no_match(self):
        from evaluation.src.evaluator import RAGEvaluator
        ev = RAGEvaluator.__new__(RAGEvaluator)
        score = ev._token_overlap("apple orange", "banana grape")
        assert score == 0.0

    def test_retrieval_hit_full(self):
        from evaluation.src.evaluator import RAGEvaluator
        ev = RAGEvaluator.__new__(RAGEvaluator)
        score = ev._retrieval_hit(
            sources          = ["loan_agreement.pdf (page 2)"],
            expected_sources = ["loan_agreement"],
        )
        assert score == 1.0

    def test_retrieval_hit_partial(self):
        from evaluation.src.evaluator import RAGEvaluator
        ev = RAGEvaluator.__new__(RAGEvaluator)
        score = ev._retrieval_hit(
            sources          = ["loan_agreement.pdf (page 2)"],
            expected_sources = ["loan_agreement", "policy.pdf"],
        )
        assert score == 0.5

    def test_faithfulness_zero_chunks(self):
        from evaluation.src.evaluator import RAGEvaluator
        ev = RAGEvaluator.__new__(RAGEvaluator)
        assert ev._faithfulness_score("some answer", 0, []) == 0.0

    def test_faithfulness_with_sources(self):
        from evaluation.src.evaluator import RAGEvaluator
        ev = RAGEvaluator.__new__(RAGEvaluator)
        score = ev._faithfulness_score("The EMI is 10234", 3, ["loan.pdf"])
        assert score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Day 3: Redis Cache
# ─────────────────────────────────────────────────────────────────────────────
class TestRedisCache:

    def test_cache_key_deterministic(self):
        from cache.src.redis_cache import RedisCache
        key1 = RedisCache._emb_key("hello world")
        key2 = RedisCache._emb_key("hello world")
        key3 = RedisCache._emb_key("different text")
        assert key1 == key2
        assert key1 != key3

    def test_query_key_includes_doc_type(self):
        from cache.src.redis_cache import RedisCache
        key1 = RedisCache._query_key("What is EMI?", "loan_agreement")
        key2 = RedisCache._query_key("What is EMI?", "policy")
        key3 = RedisCache._query_key("What is EMI?", None)
        assert key1 != key2
        assert key1 != key3

    def test_cache_disabled_when_redis_unavailable(self):
        from cache.src.redis_cache import RedisCache
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("connection refused")
            cache = RedisCache()
        assert cache._enabled is False
        assert cache.get_embedding("test") is None   # graceful None, no crash


# ─────────────────────────────────────────────────────────────────────────────
# Day 4: BFSI Extractor
# ─────────────────────────────────────────────────────────────────────────────
class TestBFSIExtractor:

    def test_parse_valid_json(self):
        from extraction.src.bfsi_extractor import BFSIExtractor
        extractor = BFSIExtractor.__new__(BFSIExtractor)
        result    = extractor._parse_json_response(
            '{"loan_id": "L-001", "interest_rate": 8.5}',
            {}
        )
        assert result["loan_id"]       == "L-001"
        assert result["interest_rate"] == 8.5

    def test_parse_json_strips_markdown_fences(self):
        from extraction.src.bfsi_extractor import BFSIExtractor
        extractor = BFSIExtractor.__new__(BFSIExtractor)
        raw       = '```json\n{"loan_id": "L-002"}\n```'
        result    = extractor._parse_json_response(raw, {})
        assert result["loan_id"] == "L-002"

    def test_parse_returns_fallback_on_invalid_json(self):
        from extraction.src.bfsi_extractor import BFSIExtractor
        extractor = BFSIExtractor.__new__(BFSIExtractor)
        result    = extractor._parse_json_response("not json at all", {"fallback": True})
        assert result == {"fallback": True}

    def test_rbi_compliance_high_rate(self):
        from extraction.src.bfsi_extractor import BFSIExtractor
        extractor  = BFSIExtractor.__new__(BFSIExtractor)
        compliance = extractor.check_rbi_compliance({"interest_rate": 40.0})
        assert compliance["compliant"] is False
        assert len(compliance["issues"]) > 0

    def test_rbi_compliance_normal_rate(self):
        from extraction.src.bfsi_extractor import BFSIExtractor
        extractor  = BFSIExtractor.__new__(BFSIExtractor)
        compliance = extractor.check_rbi_compliance({"interest_rate": 8.5})
        assert compliance["compliant"] is True
        assert len(compliance["issues"]) == 0
