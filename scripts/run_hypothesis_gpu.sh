#!/usr/bin/env bash
# One-shot GPU run: AR (none/both) vs DSBD 2x2 on the same Spider examples.
#
#   bash scripts/run_hypothesis_gpu.sh
#
# Optional env overrides:
#   NUM_INPUTS=20 MAX_TOKENS=64 GPU=0 bash scripts/run_hypothesis_gpu.sh
#   CONSTRAINT_SITE=main bash scripts/run_hypothesis_gpu.sh   # main verify only
#   CONSTRAINT_SITE=both bash scripts/run_hypothesis_gpu.sh   # draft + main
#   CONSTRAINT_SITE=draft bash scripts/run_hypothesis_gpu.sh  # draft only (default)
#
# Expected wall time (models already cached, 1x A100 / RTX 4090-class):
#   NUM_INPUTS=5   smoke     ~15–25 min
#   NUM_INPUTS=20  cheap     ~40–70 min
#   NUM_INPUTS=50  default   ~1.5–2.5 h   (up to ~3.5 h on 16–24 GB / slower GPUs)
#
# Work: 1 model load + 6 decode settings (AR none, AR both, DSBD none/xgrammar/z3/both).
# At 50 examples that is 300 generations, max_tokens=64.
#
# For draft vs main vs both in one run with separate plots, use:
#   bash scripts/run_site_ablation_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_INPUTS="${NUM_INPUTS:-50}"
MAX_TOKENS="${MAX_TOKENS:-64}"
GPU="${GPU:-0}"
APPROX="${APPROX_MODEL:-meta-llama/Llama-3.2-1B}"
TARGET="${TARGET_MODEL:-meta-llama/Llama-3.1-8B}"
CONSTRAINT_SITE="${CONSTRAINT_SITE:-draft}"
METRICS="${METRICS_CSV:-logs/hypothesis_metrics_${CONSTRAINT_SITE}.csv}"
SUMMARY="${SUMMARY_CSV:-logs/hypothesis_summary_${CONSTRAINT_SITE}.csv}"
PLOTS="${PLOT_DIR:-logs/hypothesis_plots_${CONSTRAINT_SITE}}"
LOG="${LOG_FILE:-logs/hypothesis_run_${CONSTRAINT_SITE}.txt}"
MAX_SECONDS="${MAX_SECONDS:-20000}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

case "$CONSTRAINT_SITE" in
  draft|main|both) ;;
  *)
    echo "ERROR: CONSTRAINT_SITE must be draft, main, or both (got: $CONSTRAINT_SITE)"
    exit 1
    ;;
esac

echo "=== hypothesis run ==="
echo "cwd=$ROOT  GPU=$GPU  N=$NUM_INPUTS  max_tokens=$MAX_TOKENS  site=$CONSTRAINT_SITE"
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

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]] \
  && [[ ! -f "${HOME}/.cache/huggingface/token" ]] \
  && [[ ! -f "${HOME}/.huggingface/token" ]]; then
  echo "WARNING: no HF token / huggingface login detected. Gated Llama-3 downloads will fail."
fi

mkdir -p logs "$PLOTS"
# fresh CSV so a rerun is not appended onto an old file
rm -f "$METRICS"

echo "=== decode (AR none/both + DSBD 2x2, site=${CONSTRAINT_SITE}) ==="
python evaluation.py \
  --approx_model_name "$APPROX" \
  --target_model_name "$TARGET" \
  --dataset spider \
  --num_inputs "$NUM_INPUTS" \
  --max_tokens "$MAX_TOKENS" \
  --max_seconds "$MAX_SECONDS" \
  --hypothesis_run \
  --constraint_site "$CONSTRAINT_SITE" \
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
echo "  site:    $CONSTRAINT_SITE"
echo "  log:     $LOG"
echo "  metrics: $METRICS"
echo "  summary: $SUMMARY"
echo "  plots:   $PLOTS/goodput_vs_accuracy.png"
echo "           $PLOTS/throughput_vs_goodput.png"
echo "           $PLOTS/throughput_vs_main.png"
echo
echo "Tip: compare sites with separate runs, or one ablation:"
echo "  CONSTRAINT_SITE=main bash scripts/run_hypothesis_gpu.sh"
echo "  CONSTRAINT_SITE=both bash scripts/run_hypothesis_gpu.sh"
echo "  bash scripts/run_site_ablation_gpu.sh"
