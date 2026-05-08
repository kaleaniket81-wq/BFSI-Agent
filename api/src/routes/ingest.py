"""api/src/routes/ingest.py — /api/ingest endpoint."""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.src.schemas.models import IngestResponse, DocType
from ingestion.src.pipeline import IngestionPipeline

router   = APIRouter(prefix="/api", tags=["Ingestion"])
pipeline = IngestionPipeline()

ALLOWED_TYPES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
MAX_SIZE_MB   = 50


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file:     UploadFile = File(...),
    doc_type: str        = Form("general"),
):
    """
    Upload a PDF or DOCX document for ingestion.
    Parses → chunks → embeds (Ollama local) → stores in pgvector.

    - **file**: PDF or DOCX file
    - **doc_type**: loan_agreement | policy | contract | emi_schedule | general
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Only PDF and DOCX supported. Got: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_SIZE_MB}MB.")

    suffix = ".pdf" if file.content_type == "application/pdf" else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest(tmp_path, doc_type=doc_type)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)
