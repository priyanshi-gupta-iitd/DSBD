#!/usr/bin/env bash
# One-shot GPU run: AR (none/both) vs DSBD 2x2 on the same Spider examples.
#
#   bash scripts/run_hypothesis_gpu.sh
#
# Optional env overrides:
#   NUM_INPUTS=20 MAX_TOKENS=64 GPU=0 bash scripts/run_hypothesis_gpu.sh
#
# Expected wall time (models already cached, 1x A100 / RTX 4090-class):
#   NUM_INPUTS=5   smoke     ~15–25 min
#   NUM_INPUTS=20  cheap     ~40–70 min
#   NUM_INPUTS=50  default   ~1.5–2.5 h   (up to ~3.5 h on 16–24 GB / slower GPUs)
#
# Work: 1 model load + 6 decode settings (AR none, AR both, DSBD none/xgrammar/z3/both).
# At 50 examples that is 300 generations, max_tokens=64.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_INPUTS="${NUM_INPUTS:-50}"
MAX_TOKENS="${MAX_TOKENS:-64}"
GPU="${GPU:-0}"
APPROX="${APPROX_MODEL:-meta-llama/Llama-3.2-1B}"
TARGET="${TARGET_MODEL:-meta-llama/Llama-3.1-8B}"
METRICS="${METRICS_CSV:-logs/hypothesis_metrics.csv}"
SUMMARY="${SUMMARY_CSV:-logs/hypothesis_summary.csv}"
PLOTS="${PLOT_DIR:-logs/hypothesis_plots}"
LOG="${LOG_FILE:-logs/hypothesis_run.txt}"
MAX_SECONDS="${MAX_SECONDS:-20000}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

echo "=== hypothesis run ==="
echo "cwd=$ROOT  GPU=$GPU  N=$NUM_INPUTS  max_tokens=$MAX_TOKENS"
echo "metrics=$METRICS  plots=$PLOTS"
echo "expected time: N=5 ~15-25m | N=20 ~40-70m | N=50 ~1.5-2.5h"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found; this will be extremely slow on CPU."
else
  nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader || true
fi

test -f spider/spider/dev.json \
  && test -f spider/spider/tables.json \
  && test -f spider/database/concert_singer/concert_singer.sqlite \
  || { echo "Spider layout missing. See README (need spider/spider/*.json and spider/database/)."; exit 1; }

if [[ -z "${HFTOKEN:-}" ]] && [[ ! -f "${HOME}/.cache/huggingface/token" ]] && [[ ! -f "${HOME}/.huggingface/token" ]]; then
  echo "WARNING: no HFTOKEN / huggingface login detected. Gated Llama-3 downloads will fail."
fi

mkdir -p logs "$PLOTS"
# fresh CSV so a rerun is not appended onto an old file
rm -f "$METRICS"

echo "=== decode (AR none/both + DSBD 2x2) ==="
python evaluation.py \
  --approx_model_name "$APPROX" \
  --target_model_name "$TARGET" \
  --dataset spider \
  --num_inputs "$NUM_INPUTS" \
  --max_tokens "$MAX_TOKENS" \
  --max_seconds "$MAX_SECONDS" \
  --hypothesis_run \
  --metrics_csv "$METRICS" \
  --log_file "$LOG"

echo
echo "=== aggregate + plots ==="
python scripts/analyze_constraint_results.py \
  --metrics_csv "$METRICS" \
  --out_csv "$SUMMARY" \
  --plot_dir "$PLOTS"

echo
echo "Done."
echo "  log:     $LOG"
echo "  metrics: $METRICS"
echo "  summary: $SUMMARY"
echo "  plots:   $PLOTS/goodput_vs_accuracy.png"
echo "           $PLOTS/goodput_correct_vs_accuracy.png"
