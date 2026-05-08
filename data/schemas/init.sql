-- data/schemas/init.sql
-- Runs automatically on first container start

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Documents table ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename     VARCHAR(500)  NOT NULL,
    doc_type     VARCHAR(50)   NOT NULL,   -- loan_agreement | policy | contract | emi_schedule
    file_size    INTEGER,
    page_count   INTEGER,
    status       VARCHAR(30)   DEFAULT 'uploaded',   -- uploaded | processing | indexed | failed
    uploaded_by  VARCHAR(100),
    created_at   TIMESTAMP     DEFAULT NOW(),
    updated_at   TIMESTAMP     DEFAULT NOW()
);

-- ── Document chunks with vector embeddings ───────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id  UUID          REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER       NOT NULL,
    chunk_text   TEXT          NOT NULL,
    page_number  INTEGER,
    token_count  INTEGER,
    embedding    vector(768),   -- nomic-embed-text = 768-dim | ada-002 = 1536-dim
    created_at   TIMESTAMP     DEFAULT NOW()
);

-- ── Loan records (structured data) ───────────────────────────
CREATE TABLE IF NOT EXISTS loan_records (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    loan_id         VARCHAR(50)   UNIQUE NOT NULL,
    document_id     UUID          REFERENCES documents(id),
    customer_name   VARCHAR(200),
    loan_amount     DECIMAL(15,2),
    interest_rate   DECIMAL(5,2),
    tenure_months   INTEGER,
    emi_amount      DECIMAL(12,2),
    disbursement_dt DATE,
    status          VARCHAR(30)   DEFAULT 'active',  -- active | closed | npa | overdue
    created_at      TIMESTAMP     DEFAULT NOW()
);

-- ── EMI schedule ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS emi_schedule (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    loan_id         VARCHAR(50)   REFERENCES loan_records(loan_id),
    installment_no  INTEGER,
    due_date        DATE,
    emi_amount      DECIMAL(12,2),
    principal       DECIMAL(12,2),
    interest        DECIMAL(12,2),
    paid_date       DATE,
    paid_amount     DECIMAL(12,2),
    status          VARCHAR(20)   DEFAULT 'pending'  -- pending | paid | overdue
);

-- ── Query audit log ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_log (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question     TEXT          NOT NULL,
    answer       TEXT,
    llm_provider VARCHAR(20),  -- ollama | azure
    tokens_used  INTEGER,
    latency_ms   INTEGER,
    created_at   TIMESTAMP     DEFAULT NOW()
);

-- ── Indexes for performance ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chunks_document_id  ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding     ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_loans_status         ON loan_records(status);
CREATE INDEX IF NOT EXISTS idx_emi_due_date         ON emi_schedule(due_date);
CREATE INDEX IF NOT EXISTS idx_emi_status           ON emi_schedule(status);

-- ── Sample data ───────────────────────────────────────────────
INSERT INTO loan_records (loan_id, customer_name, loan_amount, interest_rate, tenure_months, emi_amount, disbursement_dt, status)
VALUES
  ('L-2024-001', 'Rajesh Kumar',    500000.00, 8.5,  60, 10234.56, '2024-01-15', 'active'),
  ('L-2024-002', 'Priya Sharma',    250000.00, 9.0,  36,  7954.32, '2024-02-01', 'active'),
  ('L-2024-003', 'Amit Verma',     1000000.00, 7.75, 120, 12043.21, '2024-03-10', 'active'),
  ('L-2024-004', 'Sunita Patel',    150000.00, 10.5, 24,  6940.87, '2024-01-20', 'overdue'),
  ('L-2024-005', 'Vikram Singh',    750000.00, 8.25, 84, 11234.00, '2023-12-01', 'npa')
ON CONFLICT (loan_id) DO NOTHING;
