#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---full}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[Desc2Fix] Mode: $MODE"

if [[ "$MODE" == "--full" ]]; then
  echo "[1/9] Setting up benchmarks"
  python scripts/setup_benchmarks.py --all

  echo "[2/9] Preparing corpora"
  python scripts/prepare_corpus.py --all --mode raw

  echo "[3/9] Validating corpora"
  python scripts/validate_corpus.py --all

  echo "[4/9] Extracting structured bug signals"
  python scripts/extract_signals.py --all --resume

  echo "[5/9] Running RQ1"
  python scripts/run_rq1.py --all

  echo "[6/9] Running RQ2"
  python scripts/run_rq2.py --all

  echo "[7/9] Running RQ3"
  python scripts/run_rq3.py --all

  echo "[8/9] Running RQ4 internal"
  python scripts/run_rq4_internal.py --all

  echo "[9/9] Generating tables and figures"
  python scripts/generate_tables.py --all
  python scripts/generate_figures.py --all

  echo "[OK] Full reproduction complete."
  echo "Launch dashboard with:"
  echo "  streamlit run dashboard/app.py"

elif [[ "$MODE" == "--no-llm" ]]; then
  echo "[1/6] Setting up benchmarks"
  python scripts/setup_benchmarks.py --all

  echo "[2/6] Preparing corpora"
  python scripts/prepare_corpus.py --all --mode raw

  echo "[3/6] Validating corpora"
  python scripts/validate_corpus.py --all

  echo "[4/6] Running deterministic metric pipelines only"
  python scripts/run_rq1.py --all --skip-llm
  python scripts/run_rq2.py --step metric-alignment
  python scripts/run_rq3.py --step metric-alignment
  python scripts/run_rq4_internal.py --step metric-alignment

  echo "[5/6] Generating tables"
  python scripts/generate_tables.py --all

  echo "[6/6] Generating figures"
  python scripts/generate_figures.py --all

  echo "[OK] No-LLM reproduction complete."

else
  echo "Unknown mode: $MODE"
  echo "Available modes:"
  echo "  --full"
  echo "  --no-llm"
  exit 1
fi
