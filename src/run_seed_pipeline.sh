#!/usr/bin/env bash
# Runs the FULL attack + faithfulness pipeline for one target-model seed,
# reusing 100% existing, already-tested tools (attacks.build_eval_sample,
# attacks.unguided, attacks.guided, faithfulness.deletion_auc,
# attacks.compare_guided_vs_unguided, faithfulness.correlate_with_attacks).
# No new attack/analysis logic -- this only automates sequencing so you don't
# have to hand-type ~6 commands per seed across 5 seeds.
#
# Prerequisite: the target checkpoint for this seed must already exist
# (runs/runs_v1/model_seed<N>.pt), which it does for seeds 0-4 from your
# original 5-seed Table 1 run.
#
# Usage (from src/, with the venv activated):
#   caffeinate -i ./run_seed_pipeline.sh 1
#   caffeinate -i ./run_seed_pipeline.sh 2
#   ... etc for each additional seed
#
# Each seed's results land in a SEPARATE folder (results/tables/seed<N>/) so
# nothing overwrites another seed's data. Seed 0's results already exist
# directly in results/tables/ from earlier work -- run
# ./copy_seed0_results.sh once (see below) to mirror them into
# results/tables/seed0/ for a consistent cross-seed folder layout before
# building the cross-seed aggregation tables.

set -e

SEED=$1
if [ -z "$SEED" ]; then
  echo "Usage: ./run_seed_pipeline.sh <seed_number>"
  exit 1
fi

SPLIT_DIR=../data/processed/split_v1
CKPT=../runs/runs_v1/model_seed${SEED}.pt
DATA_OUT=../data/processed
RESULTS_OUT=../results/tables/seed${SEED}
ELIG_SAMPLE=${DATA_OUT}/attack_eval_sample_seed${SEED}_v1.csv

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT"
  echo "Train this seed first, e.g.:"
  echo "  python train.py --split_dir $SPLIT_DIR --outdir ../runs/runs_v1 --seeds $SEED --epochs 15 --batch_size 256"
  exit 1
fi

mkdir -p "$RESULTS_OUT"

echo "############################################################"
echo "# Seed $SEED pipeline started: $(date)"
echo "############################################################"

echo ""
echo "=== [seed $SEED] Step 1/6: this seed's own eligible-population sample ==="
echo "Started: $(date)"
python -m attacks.build_eval_sample --split_dir "$SPLIT_DIR" --checkpoint "$CKPT" \
    --outdir "$DATA_OUT" --filename "attack_eval_sample_seed${SEED}_v1.csv" \
    --n 10000 --seed 0

echo ""
echo "=== [seed $SEED] Step 2/6: unguided attacks (6 methods, n=10000) ==="
echo "Started: $(date)"
python -m attacks.unguided --split_dir "$SPLIT_DIR" --checkpoint "$CKPT" \
    --outdir "$RESULTS_OUT" --eval_sample "$ELIG_SAMPLE" \
    --max_budget 10 --budgets 1 2 3 5 8 10 --seed 0

echo ""
echo "=== [seed $SEED] Step 3/6: guided attacks (one_shot + adaptive, n=500) ==="
echo "Started: $(date)"
python -m attacks.guided --split_dir "$SPLIT_DIR" --checkpoint "$CKPT" \
    --outdir "$RESULTS_OUT" --eval_sample "$ELIG_SAMPLE" \
    --modes one_shot adaptive --max_budget 10 --budgets 1 2 3 5 8 10 \
    --ig_steps 20 --n_urls 500 --seed 0

echo ""
echo "=== [seed $SEED] Step 4/6: faithfulness scoring (n=1000) ==="
echo "Started: $(date)"
python -m faithfulness.deletion_auc --split_dir "$SPLIT_DIR" --checkpoint "$CKPT" \
    --eval_sample "$ELIG_SAMPLE" --outdir "$RESULTS_OUT" \
    --ig_steps 20 --n_urls 1000 --topk_fraction 0.2 --seed 0

echo ""
echo "=== [seed $SEED] Step 5/6: guided-vs-unguided comparison (Table 3) ==="
echo "Started: $(date)"
python -m attacks.compare_guided_vs_unguided --outdir "$RESULTS_OUT" \
    --max_budget 10 --budgets 1 2 3 5 8 10

echo ""
echo "=== [seed $SEED] Step 6/6: faithfulness-efficiency correlation (Table 4) ==="
echo "Started: $(date)"
python -m faithfulness.correlate_with_attacks --outdir "$RESULTS_OUT" \
    --n_bootstrap 2000 --seed 0

echo ""
echo "############################################################"
echo "# Seed $SEED pipeline FINISHED: $(date)"
echo "############################################################"
echo "Results in: $RESULTS_OUT/"
echo "  table2_results.csv, table3_results.csv, table4_results.csv,"
echo "  table4_by_class.csv, unguided_attack_results.csv, guided_attack_results.csv,"
echo "  faithfulness_results.csv"
