#!/usr/bin/env bash
# Quick-start script — sets up the full stack locally in ~2 minutes.
# Usage:  bash scripts/quickstart.sh [groq|ollama]
set -euo pipefail

BACKEND=${1:-groq}
ENV_FILE=".env"

echo "════════════════════════════════════════════"
echo "  Clinical RAG Platform — Local Quick Start"
echo "  LLM backend: $BACKEND"
echo "════════════════════════════════════════════"

# ── 1. Create .env from example if missing ────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  cp .env.example "$ENV_FILE"
  echo "Created .env from .env.example"
fi

# ── 2. Groq key prompt ────────────────────────────────────────────────────────
if [[ "$BACKEND" == "groq" ]]; then
  if ! grep -q "^GROQ_API_KEY=gsk_" "$ENV_FILE" 2>/dev/null; then
    echo ""
    echo "Groq is free — get a key at https://console.groq.com (takes 30 seconds)"
    echo -n "Paste your GROQ_API_KEY: "
    read -r GROQ_KEY
    sed -i.bak "s|^GROQ_API_KEY=.*|GROQ_API_KEY=$GROQ_KEY|" "$ENV_FILE"
    rm -f "$ENV_FILE.bak"
  fi
  sed -i.bak "s|^LLM_BACKEND=.*|LLM_BACKEND=groq|" "$ENV_FILE"
  rm -f "$ENV_FILE.bak"
fi

# ── 3. Ollama mode ────────────────────────────────────────────────────────────
if [[ "$BACKEND" == "ollama" ]]; then
  sed -i.bak "s|^LLM_BACKEND=.*|LLM_BACKEND=ollama|" "$ENV_FILE"
  rm -f "$ENV_FILE.bak"
  echo ""
  echo "Pulling Ollama with llama3.2 (this downloads ~2GB the first time)..."
  docker compose --profile ollama up -d ollama
  sleep 10
  docker exec clinical-ollama ollama pull llama3.2
fi

# ── 4. Start core services ────────────────────────────────────────────────────
echo ""
echo "Starting services (Qdrant + Redis + Prometheus + Grafana + API)..."
docker compose up -d --build

# ── 5. Wait for API to be healthy ────────────────────────────────────────────
echo -n "Waiting for API to be ready"
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 3
done

# ── 6. Ingest sample clinical documents ──────────────────────────────────────
echo ""
echo "Ingesting sample clinical documents..."
docker exec clinical-rag-app python scripts/ingest_sample_docs.py || \
  echo "Ingest script failed — check docker logs clinical-rag-app"

echo ""
echo "════════════════════════════════════════════"
echo "  Everything is running!"
echo ""
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Grafana:    http://localhost:3000  (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Qdrant UI:  http://localhost:6333/dashboard"
echo ""
echo "  Test a query:"
echo '  curl -X POST http://localhost:8000/api/v1/query \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"question": "What are the symptoms of sepsis?", "top_k": 3, "session_id": "demo"}'"'"
echo "════════════════════════════════════════════"
