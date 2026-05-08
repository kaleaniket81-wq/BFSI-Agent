"""api/src/routes/health.py — Health check endpoint."""
import os
import requests
import psycopg2
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    status = {"api": "ok", "postgres": "unknown", "redis": "unknown", "ollama": "unknown"}

    # Check Postgres
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME", "bfsi_intelligence"),
            user=os.getenv("DB_USER", "bfsi_user"),
            password=os.getenv("DB_PASSWORD"),
        )
        conn.close()
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"

    # Check Redis
    try:
        import redis
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
        )
        r.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"

    # Check Ollama
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            status["ollama"] = "ok"
            status["ollama_models"] = models
        else:
            status["ollama"] = f"error: status {r.status_code}"
    except Exception as e:
        status["ollama"] = f"error: {str(e)} — run 'ollama serve' on host"

    overall = "healthy" if all(v == "ok" for k, v in status.items() if k != "ollama_models") else "degraded"
    return {"status": overall, "services": status}
