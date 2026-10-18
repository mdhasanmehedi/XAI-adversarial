#!/usr/bin/env bash
# Full PhiUSIIL replication pipeline for ONE seed (run from src/, venv active).
#   caffeinate -i ./run_phiusiil_pipeline.sh 0
#   caffeinate -i ./run_phiusiil_pipeline.sh 1
#   caffeinate -i ./run_phiusiil_pipeline.sh 2
# Prerequisites (once):
#   python prep_phiusiil.py --csv ../data/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
#        --outdir ../data/processed/split_phiusiil --seed 0
#   python train.py --split_dir ../data/processed/split_phiusiil --outdir ../runs/runs_phiusiil \
#        --seeds 0 1 2 --epochs 10 --batch_size 128
# Design: one frozen 500-URL attack sample per seed is used by EVERY method, so all paired
# comparisons are on identical URLs and no further subsampling happens in any script.
set -e
SEED=$1
[ -z "$SEED" ] && { echo "usage: $0 <seed>"; exit 1; }
SPLIT=../data/processed/split_phiusiil
CKPT=../runs/runs_phiusiil/model_seed${SEED}.pt
SAMPLE=../data/processed/attack_eval_sample_seed${SEED}_phiusiil.csv
OUT=../results/phiusiil/seed${SEED}
[ -f "$CKPT" ] || { echo "missing $CKPT -- train first"; exit 1; }
mkdir -p "$OUT"

echo "== [$SEED] 1/8 frozen 500-URL sample"
python -m attacks.build_eval_sample --split_dir $SPLIT --checkpoint $CKPT \
   --outdir ../data/processed --filename attack_eval_sample_seed${SEED}_phiusiil.csv --n 500 --seed 0

echo "== [$SEED] 2/8 unguided baselines (V0)"
python -m attacks.unguided --split_dir $SPLIT --checkpoint $CKPT --outdir $OUT \
   --eval_sample $SAMPLE --max_budget 10 --budgets 1 2 3 5 8 10 --seed 0

echo "== [$SEED] 3/8 original IG guided attack (V0)"
python -m attacks.guided --split_dir $SPLIT --checkpoint $CKPT --outdir $OUT \
   --eval_sample $SAMPLE --modes one_shot adaptive --max_budget 10 --budgets 1 2 3 5 8 10 \
   --ig_steps 20 --n_urls 500 --seed 0

echo "== [$SEED] 4/8 ranking variants (V0): random, saliency, gradxinput, occlusion, ig re-run"
python -m attacks.guided_variants --split_dir $SPLIT --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random gradxinput saliency occlusion ig \
   --modes one_shot adaptive --max_budget 10 --n_urls 500 --ig_steps 20 --seed 0

echo "== [$SEED] 5/8 unguided baselines (V1, prefix frozen)"
python -m attacks.unguided_protected --checkpoint $CKPT --eval_sample $SAMPLE --outdir $OUT \
   --n_urls 500 --max_budget 10 --seed 0

echo "== [$SEED] 6/8 ranking variants (V1)"
python -m attacks.guided_variants --split_dir $SPLIT --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random gradxinput saliency occlusion ig \
   --modes one_shot adaptive --max_budget 10 --n_urls 500 --ig_steps 20 --seed 0 --protect_prefix

echo "== [$SEED] 7/8 faithfulness of each ranker (V0)"
python -m faithfulness.faithfulness_by_ranker --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random saliency gradxinput ig occlusion \
   --n_urls 500 --ig_steps 20 --topk_fraction 0.2 --seed 0

echo "== [$SEED] 8/8 faithfulness of each ranker (V1)"
python -m faithfulness.faithfulness_by_ranker --checkpoint $CKPT --eval_sample $SAMPLE \
   --outdir $OUT --rankers random saliency gradxinput ig occlusion \
   --n_urls 500 --ig_steps 20 --topk_fraction 0.2 --seed 0 --protect_prefix
echo "== seed $SEED finished: $OUT"
