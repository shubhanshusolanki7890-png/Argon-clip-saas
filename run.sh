#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/2] Building frontend production bundle..."
(cd frontend && npm run build)

echo "[2/2] Starting ArgonClip FastAPI server on http://0.0.0.0:${PORT:-8000} ..."
exec python3 -m uvicorn backend.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
