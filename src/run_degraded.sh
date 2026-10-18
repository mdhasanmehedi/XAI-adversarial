#!/usr/bin/env bash
# degraded experiment, seeds 0-4, V0 and V1. Run from src/ with .venv active:  caffeinate -i bash run_degraded.sh
set -e
SUBSEED=0
for N in 0 1 2 3 4; do
  CKPT=../runs/runs_v1/model_seed${N}.pt
  if [ "$N" = "0" ]; then SAMPLE=../data/processed/attack_eval_sample_v1.csv
  else SAMPLE=../data/processed/attack_eval_sample_seed${N}_v1.csv; fi
  OUT=../results/tables/seed${N}
  [ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
  [ -f "$SAMPLE" ] || { echo "missing $SAMPLE"; exit 1; }
  echo "== seed $N: degraded V0"; python -m attacks.degraded_attribution --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT --n_urls 500 --seed $SUBSEED
  echo "== seed $N: degraded V1"; python -m attacks.degraded_attribution --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT --n_urls 500 --seed $SUBSEED --protect_prefix
done
