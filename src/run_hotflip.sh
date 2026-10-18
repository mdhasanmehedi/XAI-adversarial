#!/usr/bin/env bash
# HotFlip-adapted baseline on the main corpus, all 5 seeds, V0 and V1.
# Run from src/ with the venv active:   bash run_hotflip.sh
# If SUBSEED differs from what you used for guided_variants.py (see README), edit it below.
set -e
for N in 1 2 3 4; do
  CKPT=../runs/runs_v1/model_seed${N}.pt
  if [ "$N" = "0" ]; then SAMPLE=../data/processed/attack_eval_sample_v1.csv
  else SAMPLE=../data/processed/attack_eval_sample_seed${N}_v1.csv; fi
  OUT=../results/tables/seed${N}
  SUBSEED=0
  [ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
  [ -f "$SAMPLE" ] || { echo "missing $SAMPLE"; exit 1; }
  echo "== seed $N: HotFlip V0";  python -m attacks.hotflip --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT --modes one_shot adaptive --max_budget 10 --n_urls 500 --seed $SUBSEED
  echo "== seed $N: HotFlip V1";  python -m attacks.hotflip --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT --modes one_shot adaptive --max_budget 10 --n_urls 500 --seed $SUBSEED --protect_prefix
done
