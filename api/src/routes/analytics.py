"""api/src/routes/analytics.py — /api/analytics endpoints."""
from fastapi import APIRouter, HTTPException, Query
from api.src.schemas.models import AnalyticsRequest, AnalyticsResponse
from analytics.src.analytics_engine import AnalyticsEngine

router = APIRouter(prefix="/api", tags=["Analytics"])
engine = AnalyticsEngine()


@router.post("/analytics/query", response_model=AnalyticsResponse)
def analytics_query(request: AnalyticsRequest):
    """
    Natural language → SQL → PostgreSQL loan database.

    Example questions:
    - "Which loans are overdue by more than 30 days?"
    - "What is the total disbursed amount by loan status?"
    - "Show top 5 customers by outstanding loan amount"
    """
    try:
        result = engine.query(request.question)
        return AnalyticsResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Analytics failed: {str(e)}")


@router.get("/analytics/portfolio-summary")
def portfolio_summary():
    """Pre-built dashboard: loan portfolio health by status."""
    try:
        return engine.portfolio_summary()
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/analytics/overdue-emi")
def overdue_emi(days: int = Query(30, ge=0, le=365)):
    """EMIs overdue by more than `days` days."""
    try:
        return engine.overdue_emi_report(days=days)
    except Exception as e:
        raise HTTPException(500, str(e))
