"""
worker/src/tasks.py

DAY 3 UPGRADE — Async Document Ingestion with Celery

Why async matters:
  Ingesting a 100-page PDF takes 60–90 seconds (parsing + embedding each chunk).
  Synchronous ingestion blocks the API thread — the user gets a timeout.
  Async ingestion: user uploads → gets job_id immediately → polls for status.

Architecture:
  POST /api/ingest  →  saves file  →  enqueues Celery task  →  returns job_id
  GET  /api/ingest/status/{job_id}  →  returns status/result

Interview line:
  "Synchronous ingestion blocked the API for 90 seconds on large documents.
   I decoupled it with Celery + Redis. The user gets a job ID in 200ms and
   can poll for completion. This is the same pattern used in payment processing
   at scale — decouple the request from the work."
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = (
    f"redis://:{os.getenv('REDIS_PASSWORD', '')}@"
    f"{os.getenv('REDIS_HOST', 'localhost')}:"
    f"{os.getenv('REDIS_PORT', 6379)}/1"
)

celery_app = Celery(
    "bfsi_worker",
    broker  = REDIS_URL,
    backend = REDIS_URL,
)

celery_app.conf.update(
    task_serializer        = "json",
    result_serializer      = "json",
    accept_content         = ["json"],
    task_track_started     = True,
    task_acks_late         = True,           # re-queue on worker crash
    worker_prefetch_multiplier = 1,          # fair dispatch for slow tasks
    result_expires         = 3600,           # keep results 1 hour
)


@celery_app.task(bind=True, name="bfsi.ingest_document", max_retries=3)
def ingest_document_task(self, filepath: str, doc_type: str, original_filename: str):
    """
    Async ingestion task.
    Retries up to 3 times with exponential backoff on failure.
    """
    try:
        # Update state so client can see progress
        self.update_state(state="PARSING", meta={"step": "parsing document"})

        from ingestion.src.parser.document_parser import DocumentParser
        parser = DocumentParser()
        parsed = parser.parse(filepath)
        parsed["doc_type"] = doc_type

        self.update_state(state="CHUNKING", meta={"step": "chunking text", "pages": parsed["page_count"]})

        from ingestion.src.chunker.smart_chunker import SmartChunker
        chunker = SmartChunker()
        chunks  = chunker.chunk_document(parsed)

        self.update_state(state="EMBEDDING", meta={
            "step": "generating embeddings",
            "total_chunks": len(chunks),
        })

        from llm.src.provider.base import LLMProviderFactory
        from cache.src.redis_cache import RedisCache, CachedLLMProvider

        base_llm = LLMProviderFactory.get()
        cache    = RedisCache()
        llm      = CachedLLMProvider(base_llm, cache)

        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = llm.embed(chunk["text"])
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                self.update_state(state="EMBEDDING", meta={
                    "step": "generating embeddings",
                    "progress": f"{i+1}/{len(chunks)}",
                })

        self.update_state(state="STORING", meta={"step": "storing in pgvector"})

        from ingestion.src.storage.vector_store import VectorStore
        import os as _os
        store  = VectorStore()
        doc_id = store.save_document(
            filename   = original_filename,
            doc_type   = doc_type,
            page_count = parsed["page_count"],
            file_size  = _os.path.getsize(filepath),
        )
        saved = store.save_chunks(doc_id, chunks, embeddings)

        # Invalidate query cache — new doc means old cached answers may be incomplete
        cache.invalidate_query_cache()

        # Clean up temp file
        try:
            _os.unlink(filepath)
        except Exception:
            pass

        return {
            "document_id":  doc_id,
            "filename":     original_filename,
            "doc_type":     doc_type,
            "pages":        parsed["page_count"],
            "chunks":       saved,
            "llm_provider": llm.name,
            "status":       "completed",
        }

    except Exception as exc:
        # Retry with exponential backoff: 30s, 60s, 120s
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
