#!/usr/bin/env python3
"""scripts/create_init_files.py — Create all __init__.py files."""
import os

paths = [
    "llm/__init__.py",
    "llm/src/__init__.py",
    "llm/src/provider/__init__.py",
    "llm/src/ollama/__init__.py",
    "ingestion/__init__.py",
    "ingestion/src/__init__.py",
    "ingestion/src/parser/__init__.py",
    "ingestion/src/chunker/__init__.py",
    "ingestion/src/storage/__init__.py",
    "rag/__init__.py",
    "rag/src/__init__.py",
    "rag/src/pipeline/__init__.py",
    "analytics/__init__.py",
    "analytics/src/__init__.py",
    "api/__init__.py",
    "api/src/__init__.py",
    "api/src/routes/__init__.py",
    "api/src/schemas/__init__.py",
    "api/src/middleware/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
]

for p in paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("")
    print(f"Created {p}")
