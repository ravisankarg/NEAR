#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NEAR_MODEL:?Set NEAR_MODEL to the downloaded EmbeddingGemma2 LiteRT checkpoint}"
NEAR_PYTHON="${NEAR_PYTHON:-python3}"
NEAR_DATA="${NEAR_DATA:-data/scifact}"
NEAR_EMBEDDINGS="${NEAR_EMBEDDINGS:-artifacts/scifact_eg2_rerun}"
NEAR_RESULTS="${NEAR_RESULTS:-results/rerun_scifact_eg2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="$PWD/artifacts/matplotlib_cache"
"$NEAR_PYTHON" -m near_retrieval.encode --data "$NEAR_DATA" --out "$NEAR_EMBEDDINGS" \
  --model "$NEAR_MODEL" --threads 8 --max-length 8192 --batch-size 16
"$NEAR_PYTHON" -m unittest discover -s tests -v
"$NEAR_PYTHON" -m near_retrieval.benchmark --data "$NEAR_DATA" --embeddings "$NEAR_EMBEDDINGS" --out "$NEAR_RESULTS"
"$NEAR_PYTHON" scripts/render_report.py "$NEAR_RESULTS/results.json"
