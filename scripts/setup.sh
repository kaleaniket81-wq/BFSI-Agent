#!/bin/bash
# ── BFSI Doc Intelligence — One-time setup script ─────────────────────────────
# Run this ONCE before docker compose up
# Requirements: Ollama installed on host (https://ollama.com/download)

set -e

echo "🔍 Checking Ollama is installed on host..."
if ! command -v ollama &> /dev/null; then
  echo "❌ Ollama not found. Install from https://ollama.com/download and re-run."
  exit 1
fi

echo "⬇️  Pulling required Ollama models (one-time download)..."
ollama pull llama3
ollama pull nomic-embed-text

echo "✅ Models ready!"
echo ""
echo "🚀 Starting project..."
docker compose down -v   # clean slate
docker compose up --build -d

echo ""
echo "✅ All services starting!"
echo "   API:      http://localhost:8000/docs"
echo "   Frontend: http://localhost:3000"
echo "   Gateway:  http://localhost:9090"
echo "   pgAdmin:  http://localhost:5050  (admin@bfsi.local / admin)"
