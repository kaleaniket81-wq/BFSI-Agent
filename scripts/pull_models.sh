#!/bin/bash
# scripts/pull_models.sh — Pull required Ollama models
# Run this once after starting Ollama: ./scripts/pull_models.sh

OLLAMA_URL=${OLLAMA_BASE_URL:-http://localhost:11434}

echo "Pulling Ollama models from $OLLAMA_URL ..."

echo "1/2 Pulling llama3 (generation model ~4.7GB)..."
curl -s "$OLLAMA_URL/api/pull" -d '{"name":"llama3"}' | tail -1

echo "2/2 Pulling nomic-embed-text (embedding model ~274MB)..."
curl -s "$OLLAMA_URL/api/pull" -d '{"name":"nomic-embed-text"}' | tail -1

echo ""
echo "Done! Verify with:"
echo "  curl $OLLAMA_URL/api/tags"
