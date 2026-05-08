-- data/schemas/migration_v2.sql
-- Run after init.sql to add Day 1 hybrid search indexes
-- psql -h <host> -U bfsi_user -d bfsi_intelligence -f data/schemas/migration_v2.sql

-- Full-text search index for BM25-style keyword retrieval (Day 1)
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON document_chunks
    USING gin(to_tsvector('english', chunk_text));

-- Chunking strategy column (Day 1 — smart chunker)
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS strategy VARCHAR(20) DEFAULT 'fixed';

-- Rewritten query logging (Day 1 — query rewriter)
ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS rewritten_query TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS rewrite_ms     INTEGER;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS retrieval_ms   INTEGER;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS llm_ms         INTEGER;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS cache_hit      BOOLEAN DEFAULT FALSE;

-- Evaluation results table (Day 2)
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_at              TIMESTAMP DEFAULT NOW(),
    total_questions     INTEGER,
    avg_answer_sim      DECIMAL(5,4),
    avg_retrieval_hit   DECIMAL(5,4),
    avg_faithfulness    DECIMAL(5,4),
    avg_latency_ms      INTEGER,
    p95_latency_ms      INTEGER,
    llm_provider        VARCHAR(20),
    chunking_strategy   VARCHAR(20),
    notes               TEXT
);

-- Async job tracking (Day 3 — Celery)
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id      VARCHAR(100) PRIMARY KEY,
    filename    VARCHAR(500),
    doc_type    VARCHAR(50),
    status      VARCHAR(30) DEFAULT 'queued',
    document_id UUID,
    error       TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- Extraction results (Day 4 — BFSI extractor)
CREATE TABLE IF NOT EXISTS extractions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID REFERENCES documents(id) ON DELETE CASCADE,
    loan_id         VARCHAR(50),
    borrower_name   VARCHAR(200),
    loan_amount     DECIMAL(15,2),
    interest_rate   DECIMAL(5,2),
    tenure_months   INTEGER,
    emi_amount      DECIMAL(12,2),
    penalties_json  JSONB,
    raw_extraction  JSONB,
    extracted_at    TIMESTAMP DEFAULT NOW()
);

-- Index on extractions for fast loan lookup
CREATE INDEX IF NOT EXISTS idx_extractions_loan_id ON extractions(loan_id);
CREATE INDEX IF NOT EXISTS idx_extractions_doc_id  ON extractions(document_id);

-- Performance view — latency breakdown per day
CREATE OR REPLACE VIEW query_performance AS
SELECT
    DATE(created_at)        AS query_date,
    llm_provider,
    COUNT(*)                AS total_queries,
    AVG(latency_ms)         AS avg_total_ms,
    AVG(rewrite_ms)         AS avg_rewrite_ms,
    AVG(retrieval_ms)       AS avg_retrieval_ms,
    AVG(llm_ms)             AS avg_llm_ms,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS cache_hit_rate
FROM query_log
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at), llm_provider
ORDER BY query_date DESC;
