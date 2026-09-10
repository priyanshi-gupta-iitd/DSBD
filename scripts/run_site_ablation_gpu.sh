#!/usr/bin/env bash
# Site ablation: constraints on main vs both (DSBD 2x2 + AR baselines).
# Writes one metrics CSV and separate plot folders per site.
#
#   bash scripts/run_site_ablation_gpu.sh
#
# Optional env overrides:
#   NUM_INPUTS=20 GPU=0 bash scripts/run_site_ablation_gpu.sh
#   SITES=draft,main,both bash scripts/run_site_ablation_gpu.sh  # include draft
#   SITES=main bash scripts/run_site_ablation_gpu.sh               # main only
#
# Expected wall time (weights cached, A100/4090-class):
#   N=20 × main,both  ~ 1.5–3 h
#   N=50 × main,both  ~ 3–6 h
#
# Plots:
#   logs/site_ablation_plots/site_main/
#   logs/site_ablation_plots/site_both/
#   logs/site_ablation_plots/goodput_vs_accuracy.png  (combined)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_INPUTS="${NUM_INPUTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-64}"
GPU="${GPU:-0}"
APPROX="${APPROX_MODEL:-meta-llama/Llama-3.2-1B}"
TARGET="${TARGET_MODEL:-meta-llama/Llama-3.1-8B}"
SITES="${SITES:-main,both}"
METRICS="${METRICS_CSV:-logs/site_ablation_metrics.csv}"
SUMMARY="${SUMMARY_CSV:-logs/site_ablation_summary.csv}"
PLOTS="${PLOT_DIR:-logs/site_ablation_plots}"
LOG="${LOG_FILE:-logs/site_ablation_run.txt}"
MAX_SECONDS="${MAX_SECONDS:-40000}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

# validate sites
IFS=',' read -r -a SITE_ARR <<< "$SITES"
for s in "${SITE_ARR[@]}"; do
  s="$(echo "$s" | xargs)"
  case "$s" in
    draft|main|both) ;;
    *)
      echo "ERROR: invalid site '$s' (allowed: draft,main,both)"
      exit 1
      ;;
  esac
done
SITE_CSV=$(IFS=,; echo "${SITE_ARR[*]}" | tr -d ' ')

N_SITES=${#SITE_ARR[@]}
echo "=== constraint site ablation ==="
echo "cwd=$ROOT  GPU=$GPU  N=$NUM_INPUTS  max_tokens=$MAX_TOKENS"
echo "sites=$SITE_CSV  ($N_SITES site(s))"
echo "metrics=$METRICS  plots=$PLOTS"
echo "expected: ~$((N_SITES))× DSBD 2x2 + AR; N=20 ~$((N_SITES))–$((N_SITES*2))h-ish"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found; this will be extremely slow on CPU."
else
  nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader || true
fi

test -f spider/spider/dev.json \
  && test -f spider/spider/tables.json \
  && test -f spider/database/concert_singer/concert_singer.sqlite \
  || { echo "Spider layout missing. See README."; exit 1; }

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]] \
  && [[ ! -f "${HOME}/.cache/huggingface/token" ]] \
  && [[ ! -f "${HOME}/.huggingface/token" ]]; then
  echo "WARNING: no HF token / huggingface login detected. Gated Llama-3 downloads will fail."
fi

mkdir -p logs "$PLOTS"
rm -f "$METRICS"

echo "=== decode (AR none/both + DSBD 2x2 × sites=${SITE_CSV}) ==="
python evaluation.py \
  --approx_model_name "$APPROX" \
  --target_model_name "$TARGET" \
  --dataset spider \
  --num_inputs "$NUM_INPUTS" \
  --max_tokens "$MAX_TOKENS" \
  --max_seconds "$MAX_SECONDS" \
  --hypothesis_run \
  --site_ablation \
  --constraint_sites "$SITE_CSV" \
  --metrics_csv "$METRICS" \
  --log_file "$LOG"

echo
echo "=== aggregate + per-site plots ==="
python scripts/analyze_constraint_results.py \
  --metrics_csv "$METRICS" \
  --out_csv "$SUMMARY" \
  --plot_dir "$PLOTS"

echo
echo "Done."
echo "  sites:   $SITE_CSV"
echo "  log:     $LOG"
echo "  metrics: $METRICS"
echo "  summary: $SUMMARY"
echo "  plots:"
for s in "${SITE_ARR[@]}"; do
  s="$(echo "$s" | xargs)"
  echo "    $PLOTS/site_${s}/goodput_vs_accuracy.png"
done
echo "    $PLOTS/goodput_vs_accuracy.png  (all sites combined)"
echo
echo "Examples:"
echo "  SITES=main,both NUM_INPUTS=20 bash scripts/run_site_ablation_gpu.sh"
echo "  CONSTRAINT_SITE=main bash scripts/run_hypothesis_gpu.sh"
