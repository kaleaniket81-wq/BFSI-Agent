"""api/src/main.py — FastAPI application with v1 + v2 routes."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.src.routes.ingest    import router as ingest_router
from api.src.routes.query     import router as query_router
from api.src.routes.analytics import router as analytics_router
from api.src.routes.health    import router as health_router
from api.src.routes.query_v2  import router as query_v2_router
from api.src.routes.ingest_v2 import router as ingest_v2_router

app = FastAPI(
    title="BFSI Document Intelligence API",
    description=(
        "On-premise AI for BFSI.\n\n"
        "v1: basic RAG\n\n"
        "v2: hybrid search + query rewriting + async ingestion + Redis cache + BFSI extraction"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(analytics_router)
app.include_router(query_v2_router)
app.include_router(ingest_v2_router)
