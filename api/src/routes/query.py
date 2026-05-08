"""api/src/routes/query.py — /api/query RAG endpoint."""
from fastapi import APIRouter, HTTPException
from api.src.schemas.models import QueryRequest, QueryResponse
from rag.src.pipeline.rag_pipeline import RAGPipeline
import traceback

router   = APIRouter(prefix="/api", tags=["Query"])
pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Ask a natural-language question about ingested BFSI documents.

    The RAG pipeline:
    1. Embeds the question locally via Ollama (nomic-embed-text)
    2. Retrieves top-k similar chunks from pgvector
    3. Generates a grounded answer via Llama 3

    Set LLM_PROVIDER=bedrock to use AWS Bedrock Claude 3 instead.

    - **question**: e.g. "What is the interest rate for loan L-2024-003?"
    - **doc_type**: optional filter — loan_agreement | policy | contract
    - **top_k**: number of chunks to retrieve (default 5)
    """
    try:
        result = pipeline.answer(
            question=request.question,
            doc_type=request.doc_type,
            top_k=request.top_k,
        )
        return QueryResponse(**result)
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[Query] ERROR: {error_detail}")

        err_str = str(e).lower()

        # Ollama not reachable
        if "connection refused" in err_str or "connectionerror" in err_str or "timeout" in err_str:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not reachable. Make sure Ollama is running on your host: run 'ollama serve' in CMD."
            )

        # Model not pulled
        if "model" in err_str and ("not found" in err_str or "pull" in err_str):
            raise HTTPException(
                status_code=503,
                detail="LLM model not found. Run: ollama pull llama3 && ollama pull nomic-embed-text"
            )

        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
