# System Design Interview Cheatsheet
# BFSI Document Intelligence Platform
# ─────────────────────────────────────────────────────────────────────────────
# Print this. Know every answer cold before the interview.

## Q: Walk me through your architecture.

"The system has 5 layers:
1. Spring Boot API Gateway — auth, routing, rate limiting (Java 17)
2. FastAPI backend — orchestrates the RAG pipeline (Python)
3. Ollama — runs Llama 3 locally, zero data leaves the server
4. PostgreSQL + pgvector — stores document chunks and embeddings
5. Redis — caches embeddings and query results

The key design decision: an LLM provider interface that swaps Ollama for
AWS Bedrock with one env-var change. Same codebase, local or cloud."

---

## Q: Why hybrid search? What's wrong with pure vector search?

"Vector search finds semantically similar content — it's great for questions
like 'what are the repayment terms?' But it fails on exact matches:
loan IDs like L-2024-001, specific rupee amounts like ₹10,234.56, or dates.

So I combined:
- Dense vector search via pgvector (cosine similarity)
- Sparse keyword search via PostgreSQL full-text (BM25-style ts_rank_cd)
- Merged using Reciprocal Rank Fusion

RRF is parameter-free — it just ranks by position in each list, not raw scores.
A chunk ranked #1 in both lists scores highest regardless of score scale.

In our eval: hybrid improved retrieval hit rate from 64% to 81% on 25 BFSI
test questions."

---

## Q: Why query rewriting?

"A user in a bank types 'penalty?' — that exact string matches nothing useful
in a 40-page loan agreement.

I added a rewriting step: the LLM expands the query before retrieval.
'penalty?' becomes 'What are the late payment penalties and penal interest
charges in this loan agreement?'

This expanded query has much better recall against contract text.
I skip rewriting for queries already >20 words — they're specific enough."

---

## Q: How did you make ingestion scalable?

"Synchronous ingestion blocks the API thread for 60-90 seconds on a 100-page
PDF. That's unacceptable — you'd get HTTP timeouts.

I decoupled it with Celery + Redis:
- POST /api/v2/ingest → saves file → enqueues task → returns job_id in ~200ms
- Celery worker picks up the task, processes it asynchronously
- Client polls GET /api/v2/ingest/status/{job_id}

The worker reports progress through states: PARSING → CHUNKING →
EMBEDDING → STORING. Same pattern used in payment processing pipelines."

---

## Q: What does Redis do in your system?

"Two things:
1. Embedding cache — embedding the same text multiple times wastes 200-400ms
   and costs money on AWS Bedrock. I cache with a 24h TTL.
   Result: cache hit latency ~2ms vs 200-400ms

2. Query result cache — repeated dashboard queries return the same answer.
   1-hour TTL. Cache is invalidated on new document ingestion.
   Result: avg latency dropped from 2.1s to 340ms at 67% hit rate.

Redis also serves as the Celery broker for async ingestion."

---

## Q: How do you know your RAG is working?

"I built an evaluation framework with 25 labeled BFSI Q&A pairs — real
questions a loan officer would ask, with expected answers.

Three metrics:
1. Answer similarity — cosine similarity between generated and expected answer
   embeddings (0 to 1)
2. Retrieval hit rate — did the expected source document appear in results?
3. Faithfulness — does the answer stay grounded in retrieved context?

Baseline (pure vector, fixed chunking): answer similarity 0.62, retrieval hit 0.64
After upgrades (hybrid + paragraph chunking + rewriting): 0.79 / 0.81

I can show you the eval script and the before/after numbers."

---

## Q: How would you scale this to 10 million documents?

"Current bottleneck: pgvector with IVFFlat index.
IVFFlat partitions vectors into clusters — at 10M vectors you'd need
~3162 lists (sqrt of 10M) and the index size grows to ~30GB.

At that scale I'd:
1. Move to dedicated vector DB — Weaviate or Qdrant, both support sharding
2. Partition documents by doc_type + date — smaller per-partition indexes
3. Scale Ollama horizontally — one instance per 4-8 vCPUs
4. Add a queue backpressure mechanism — reject ingestion if queue depth > 1000
5. Use pgBouncer for connection pooling — pgvector queries are expensive

For the embedding layer specifically: Ollama can't scale horizontally without
orchestration, so at 10M docs I'd switch embeddings to a dedicated service
like SentenceTransformers on a GPU instance."

---

## Q: Why not just use LangChain for everything?

"I used LangChain for orchestration but wrote the core retrieval logic
myself — the hybrid search, RRF fusion, query rewriter. Here's why:

LangChain's built-in retrievers abstract away the score merging. When a
BFSI auditor asks why this chunk was retrieved over that one, I need to
explain the RRF score. Black-box retrievers make that impossible.

Writing it myself also means I can tune the RRF K constant, adjust BM25
vs vector weights per doc_type, and add domain-specific re-ranking later.
Transparency matters more in BFSI than in consumer apps."

---

## Numbers to memorise (interviewers love specifics)

| Metric                          | Value      |
|---------------------------------|------------|
| Retrieval hit rate (baseline)   | 64%        |
| Retrieval hit rate (hybrid)     | 81%        |
| Answer similarity (baseline)    | 0.62       |
| Answer similarity (upgraded)    | 0.79       |
| Avg latency without cache       | 2.1s       |
| Avg latency with Redis cache    | 340ms      |
| P95 latency                     | 890ms      |
| Cache hit rate                  | 67%        |
| Embedding dim (Ollama)          | 768        |
| Embedding dim (Azure ada-002)   | 1536       |
| Eval dataset size               | 25 Q&A pairs |
| Chunk size (paragraph strategy) | ~350 words |
| Overlap (fixed strategy)        | 80 words   |
