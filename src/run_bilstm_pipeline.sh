#!/usr/bin/env bash
# Second-architecture (BiLSTM) replication of the core experiment on the MAIN corpus, ONE seed per call.
# Run from src/ with the venv active:
#   caffeinate -i bash run_bilstm_pipeline.sh 0
#   caffeinate -i bash run_bilstm_pipeline.sh 1
#   caffeinate -i bash run_bilstm_pipeline.sh 2
# Prerequisite (once): python apply_bilstm_patch.py (repo root), then train the three BiLSTM models.
# Same design as run_gram_pipeline.sh: one frozen 500-URL sample per seed used by every method.
set -e
SEED=$1
[ -z "$SEED" ] && { echo "usage: $0 <seed>"; exit 1; }
SPLIT=../data/processed/split_v1
CKPT=../runs/runs_bilstm/model_seed${SEED}.pt
SAMPLE=../data/processed/attack_eval_sample_seed${SEED}_bilstm.csv
OUT=../results/bilstm/seed${SEED}
[ -f "$CKPT" ] || { echo "missing $CKPT -- train first"; exit 1; }
mkdir -p "$OUT"
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== [$SEED] 1/10 frozen 500-URL sample"
python -m attacks.build_eval_sample --split_dir $SPLIT --checkpoint $CKPT \
   --outdir ../data/processed --filename attack_eval_sample_seed${SEED}_bilstm.csv --n 500 --seed 0

echo "== [$SEED] 2/10 unguided baselines (V0)"
python -m attacks.unguided --split_dir $SPLIT --checkpoint $CKPT --outdir $OUT \
   --eval_sample $SAMPLE --max_budget 10 --budgets 1 2 3 5 8 10 --seed 0

echo "== [$SEED] 3/10 original IG guided attack (V0)"
python -m attacks.guided --split_dir $SPLIT --checkpoint $CKPT --outdir $OUT \
   --eval_sample $SAMPLE --modes one_shot adaptive --max_budget 10 --budgets 1 2 3 5 8 10 \
   --ig_steps 20 --n_urls 500 --seed 0

echo "== [$SEED] 4/10 ranking variants (V0): random, saliency, gradxinput, occlusion, ig re-run"
python -m attacks.guided_variants --split_dir $SPLIT --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random gradxinput saliency occlusion ig \
   --modes one_shot adaptive --max_budget 10 --n_urls 500 --ig_steps 20 --seed 0

echo "== [$SEED] 5/10 unguided baselines (V1, prefix frozen)"
python -m attacks.unguided_protected --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT \
   --n_urls 500 --max_budget 10 --seed 0

echo "== [$SEED] 6/10 ranking variants (V1)"
python -m attacks.guided_variants --split_dir $SPLIT --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random gradxinput saliency occlusion ig \
   --modes one_shot adaptive --max_budget 10 --n_urls 500 --ig_steps 20 --seed 0 --protect_prefix

echo "== [$SEED] 7/10 faithfulness of each ranker (V0)"
python -m faithfulness.faithfulness_by_ranker --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random saliency gradxinput ig occlusion \
   --n_urls 500 --ig_steps 20 --topk_fraction 0.2 --seed 0

echo "== [$SEED] 8/10 faithfulness of each ranker (V1)"
python -m faithfulness.faithfulness_by_ranker --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random saliency gradxinput ig occlusion \
   --n_urls 500 --ig_steps 20 --topk_fraction 0.2 --seed 0 --protect_prefix

echo "== [$SEED] 9/10 HotFlip benchmark (V0)"
python -m attacks.hotflip --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT \
   --modes one_shot adaptive --max_budget 10 --n_urls 500 --seed 0

echo "== [$SEED] 10/10 HotFlip benchmark (V1)"
python -m attacks.hotflip --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT \
   --modes one_shot adaptive --max_budget 10 --n_urls 500 --seed 0 --protect_prefix
echo "== seed $SEED finished: $OUT"
