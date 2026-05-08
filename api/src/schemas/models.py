"""api/src/schemas/models.py — Request/response Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum


class DocType(str, Enum):
    loan_agreement = "loan_agreement"
    policy         = "policy"
    contract       = "contract"
    emi_schedule   = "emi_schedule"
    general        = "general"


class QueryRequest(BaseModel):
    question: str          = Field(..., min_length=5, example="What is the EMI for loan L-2024-001?")
    doc_type: Optional[str] = Field(None, example="loan_agreement")
    top_k:    int           = Field(5, ge=1, le=20)


class AnalyticsRequest(BaseModel):
    question: str = Field(..., example="Which loans are overdue by more than 30 days?")


class IngestResponse(BaseModel):
    document_id:  str
    filename:     str
    doc_type:     str
    pages:        int
    chunks:       int
    llm_provider: str
    elapsed_sec:  float


class QueryResponse(BaseModel):
    answer:      str
    sources:     List[str]
    chunks_used: int
    provider:    str
    latency_ms:  int


class AnalyticsResponse(BaseModel):
    success:    bool
    question:   Optional[str]
    sql:        Optional[str]
    columns:    List[str]    = []
    results:    List[Any]    = []
    row_count:  int          = 0
    provider:   Optional[str]
    latency_ms: Optional[int]
    error:      Optional[str]
