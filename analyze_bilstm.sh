#!/usr/bin/env bash
# Run once all three seeds finished (from the repository root, NOT src/).
set -e
D="results/bilstm/seed0 results/bilstm/seed1 results/bilstm/seed2"
mkdir -p results/bilstm/cross_seed
python src/analysis/ranking_control_comparison.py --seed_dirs $D --budget 10 --base ig_original \
   --out results/bilstm/cross_seed/ranking_control.csv
python src/analysis/ranking_control_comparison.py --seed_dirs $D --budget 10 --base ig \
   --out results/bilstm/cross_seed/ranking_control_igrerun.csv
python src/analysis/protected_prefix_comparison.py --seed_dirs $D --out results/bilstm/cross_seed/protected_prefix.csv
python src/analysis/protected_vs_unguided.py --seed_dirs $D --out results/bilstm/cross_seed/protected_vs_unguided.csv
python src/analysis/faithfulness_vs_efficiency.py --seed_dirs $D --budget 10 \
   --out results/bilstm/cross_seed/faithfulness_vs_efficiency.csv
python src/analysis/faithfulness_vs_efficiency.py --seed_dirs $D --budget 10 --protected \
   --out results/bilstm/cross_seed/faithfulness_vs_efficiency_protected.csv
python src/analysis/analyze_hotflip.py --seed_dirs $D --out results/bilstm/cross_seed/hotflip_vs_rankers.csv
python src/analysis/analyze_hotflip.py --protected --seed_dirs $D --out results/bilstm/cross_seed/hotflip_vs_rankers_protected.csv
