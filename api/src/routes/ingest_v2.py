"""
api/src/routes/ingest_v2.py — Async ingestion route (Day 3 upgrade).

POST /api/v2/ingest    → enqueue Celery task → return job_id immediately
GET  /api/v2/ingest/status/{job_id} → poll task status
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2", tags=["Ingest v2 — Async"])

ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_SIZE_MB = 50


@router.post("/ingest")
async def ingest_async(
    file:     UploadFile = File(...),
    doc_type: str        = Form("general"),
    strategy: str        = Form("paragraph"),   # fixed | paragraph | semantic
):
    """
    Async document ingestion.
    Returns job_id immediately — poll /api/v2/ingest/status/{job_id} for result.

    Chunking strategies:
    - paragraph (default) — best for loan agreements and contracts
    - fixed     — fastest, good for generic documents
    - semantic  — slowest, best quality (requires Ollama)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Only PDF and DOCX supported. Got: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_SIZE_MB}MB.")

    # Save temp file (worker picks it up)
    suffix = ".pdf" if "pdf" in file.content_type else ".docx"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(contents)
    tmp.close()

    # Set chunking strategy via env (worker reads it)
    os.environ["CHUNKING_STRATEGY"] = strategy

    try:
        from worker.src.tasks import ingest_document_task
        task = ingest_document_task.delay(
            filepath          = tmp.name,
            doc_type          = doc_type,
            original_filename = file.filename,
        )
        return {
            "job_id":   task.id,
            "status":   "queued",
            "filename": file.filename,
            "doc_type": doc_type,
            "strategy": strategy,
            "poll_url": f"/api/v2/ingest/status/{task.id}",
            "message":  "Document queued for processing. Poll poll_url for status.",
        }
    except Exception as e:
        os.unlink(tmp.name)
        # If Celery/Redis unavailable, fall back to synchronous processing
        if "redis" in str(e).lower() or "connect" in str(e).lower():
            return await _sync_fallback(tmp.name, doc_type, file.filename)
        raise HTTPException(500, f"Failed to queue task: {str(e)}")


@router.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    """Poll ingestion job status."""
    try:
        from worker.src.tasks import celery_app
        task = celery_app.AsyncResult(job_id)

        if task.state == "PENDING":
            return {"job_id": job_id, "status": "pending",   "progress": None}
        elif task.state in ("PARSING", "CHUNKING", "EMBEDDING", "STORING"):
            return {"job_id": job_id, "status": "processing", "progress": task.info}
        elif task.state == "SUCCESS":
            return {"job_id": job_id, "status": "completed",  "result": task.result}
        elif task.state == "FAILURE":
            return {"job_id": job_id, "status": "failed",     "error": str(task.info)}
        else:
            return {"job_id": job_id, "status": task.state}
    except Exception as e:
        raise HTTPException(500, f"Could not fetch task status: {str(e)}")


async def _sync_fallback(filepath: str, doc_type: str, filename: str) -> dict:
    """Synchronous fallback when Redis/Celery is unavailable."""
    try:
        from ingestion.src.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        result   = pipeline.ingest(filepath, doc_type=doc_type)
        result["async"] = False
        result["note"]  = "Processed synchronously (Celery unavailable)"
        return result
    except Exception as e:
        raise HTTPException(500, f"Sync fallback also failed: {str(e)}")
