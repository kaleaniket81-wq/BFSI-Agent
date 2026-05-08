"""tests/integration/test_api_routes.py — FastAPI integration tests with mocked backends."""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Patch all external dependencies before importing app
with patch("psycopg2.connect"), \
     patch("requests.post"), \
     patch("requests.get"):
    from api.src.main import app

client = TestClient(app)


class TestHealthRoute:

    def test_health_returns_200(self):
        with patch("psycopg2.connect") as mock_conn, \
             patch("requests.get") as mock_get:
            mock_conn.return_value.__enter__ = MagicMock()
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}]}

            resp = client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "api"      in data
            assert "provider" in data


class TestIngestRoute:

    def test_ingest_rejects_txt_files(self):
        resp = client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"some text", "text/plain")},
            data={"doc_type": "general"},
        )
        assert resp.status_code == 400

    def test_ingest_accepts_pdf(self):
        mock_result = {
            "document_id":  "abc-123",
            "filename":     "loan.pdf",
            "doc_type":     "loan_agreement",
            "pages":        5,
            "chunks":       42,
            "llm_provider": "ollama",
            "elapsed_sec":  1.5,
        }
        with patch("api.src.routes.ingest.pipeline") as mock_pipeline:
            mock_pipeline.ingest.return_value = mock_result
            # Minimal valid PDF bytes (just header)
            pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"
            resp = client.post(
                "/api/ingest",
                files={"file": ("loan.pdf", pdf_bytes, "application/pdf")},
                data={"doc_type": "loan_agreement"},
            )
        assert resp.status_code == 200
        assert resp.json()["document_id"] == "abc-123"


class TestQueryRoute:

    def test_query_returns_answer(self):
        mock_result = {
            "answer":      "The EMI for loan L-2024-001 is ₹10,234.56",
            "sources":     ["loan_agreement.pdf (page 2)"],
            "chunks_used": 3,
            "provider":    "ollama",
            "latency_ms":  520,
        }
        with patch("api.src.routes.query.pipeline") as mock_pipeline:
            mock_pipeline.answer.return_value = mock_result
            resp = client.post("/api/query", json={
                "question": "What is the EMI for loan L-2024-001?",
                "doc_type": "loan_agreement",
                "top_k":    5,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer"   in data
        assert "sources"  in data
        assert "provider" in data

    def test_query_requires_question(self):
        resp = client.post("/api/query", json={"question": "hi"})
        # "hi" is 2 chars < min_length=5
        assert resp.status_code == 422


class TestAnalyticsRoute:

    def test_portfolio_summary_returns_data(self):
        mock_data = {
            "columns": ["status", "loan_count", "total_disbursed"],
            "results": [{"status": "active", "loan_count": 3, "total_disbursed": 1750000}]
        }
        with patch("api.src.routes.analytics.engine") as mock_engine:
            mock_engine.portfolio_summary.return_value = mock_data
            resp = client.get("/api/analytics/portfolio-summary")
        assert resp.status_code == 200

    def test_analytics_nl_query(self):
        mock_result = {
            "success":    True,
            "question":   "Show overdue loans",
            "sql":        "SELECT * FROM loan_records WHERE status='overdue'",
            "columns":    ["loan_id", "customer_name"],
            "results":    [{"loan_id": "L-2024-004", "customer_name": "Sunita Patel"}],
            "row_count":  1,
            "provider":   "ollama",
            "latency_ms": 800,
        }
        with patch("api.src.routes.analytics.engine") as mock_engine:
            mock_engine.query.return_value = mock_result
            resp = client.post("/api/analytics/query",
                               json={"question": "Show all overdue loans"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
