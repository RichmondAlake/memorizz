#!/bin/zsh
# Paired A/B LongMemEval run (Oracle AI DB), kill-tolerant via checkpoints.
# Usage: OPENAI_API_KEY=... BASELINE_SRC=/path/to/baseline/src ./run_ab.sh
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
COMMON=(--num_samples 50 --context_window_tokens 4000 --disable_auto_summaries)

echo "[run_ab] arm 1: baseline (HEAD) $(date)"
PYTHONPATH="$BASELINE_SRC" $PY evaluate_memorizz.py "${COMMON[@]}" \
  --config_label baseline-HEAD \
  --checkpoint results/ab_baseline_50.ckpt.jsonl \
  --output_filename ab_baseline_50.json >> results/run_ab.log 2>&1

echo "[run_ab] arm 2: candidate (working tree) $(date)"
$PY evaluate_memorizz.py "${COMMON[@]}" \
  --config_label candidate-context-efficiency \
  --checkpoint results/ab_candidate_50.ckpt.jsonl \
  --output_filename ab_candidate_50.json >> results/run_ab.log 2>&1

echo "[run_ab] comparison $(date)"
$PY compare_runs.py results/ab_baseline_50.json results/ab_candidate_50.json \
  > results/ab_comparison.txt 2>&1
echo "[run_ab] DONE $(date)" >> results/run_ab.log
touch results/AB_DONE
