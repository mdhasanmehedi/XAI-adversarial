#!/usr/bin/env bash
# Run once all seeds finished (from the repository root, NOT src/).
set -e
D="results/phiusiil/seed0 results/phiusiil/seed1 results/phiusiil/seed2"
mkdir -p results/phiusiil/cross_seed
python src/analysis/ranking_control_comparison.py --seed_dirs $D --budget 10 --base ig_original \
   --out results/phiusiil/cross_seed/ranking_control.csv
python src/analysis/ranking_control_comparison.py --seed_dirs $D --budget 10 --base ig \
   --out results/phiusiil/cross_seed/ranking_control_igrerun.csv
python src/analysis/protected_prefix_comparison.py --seed_dirs $D --out results/phiusiil/cross_seed/protected_prefix.csv
python src/analysis/protected_vs_unguided.py --seed_dirs $D --out results/phiusiil/cross_seed/protected_vs_unguided.csv
python src/analysis/faithfulness_vs_efficiency.py --seed_dirs $D --budget 10 \
   --out results/phiusiil/cross_seed/faithfulness_vs_efficiency.csv
python src/analysis/faithfulness_vs_efficiency.py --seed_dirs $D --budget 10 --protected \
   --out results/phiusiil/cross_seed/faithfulness_vs_efficiency_protected.csv
