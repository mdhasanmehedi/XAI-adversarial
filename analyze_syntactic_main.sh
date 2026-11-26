#!/usr/bin/env bash
# Cross-seed analysis of the main-corpus results under the syntactic predicate
# (results/tables/): Tables S16-S18, S29-S32, S36-S38 and the syntactic column of S40.
# Run from the repository root:   bash analyze_syntactic_main.sh
# Every output is rewritten in place and matches the stored files
# (up to the last floating-point digit).
set -e
T=results/tables
S="$T/seed0 $T/seed1 $T/seed2 $T/seed3 $T/seed4"
O=$T/cross_seed
mkdir -p $O
export URL_VALIDITY_MODE=syntactic

# per seed: guided-vs-unguided comparison and the within-method faithfulness analysis (H6)
for s in 0 1 2 3 4; do
  python3 src/attacks/compare_guided_vs_unguided.py --outdir $T/seed$s
  python3 src/faithfulness/correlate_with_attacks.py --outdir $T/seed$s
done

# cross-seed tables
python3 src/aggregate_cross_seed.py --results_base $T --outdir $O
python3 src/analysis/ranking_control_comparison.py --seed_dirs $S --base ig_original --out $O/ranking_control.csv
python3 src/analysis/ranking_control_comparison.py --seed_dirs $S --base ig --out $O/ranking_control_igrerun.csv
python3 src/analysis/protected_prefix_comparison.py --seed_dirs $S --out $O/protected_prefix.csv
python3 src/analysis/protected_vs_unguided.py --seed_dirs $S --out $O/protected_vs_unguided.csv
python3 src/analysis/faithfulness_vs_efficiency.py --seed_dirs $S --out $O/faithfulness_vs_efficiency.csv
python3 src/analysis/faithfulness_vs_efficiency.py --protected --seed_dirs $S --out $O/faithfulness_vs_efficiency_protected.csv
python3 src/analysis/analyze_hotflip.py --seed_dirs $S --out $O/hotflip_vs_rankers.csv
python3 src/analysis/analyze_hotflip.py --protected --seed_dirs $S --out $O/hotflip_vs_rankers_protected.csv
python3 src/analysis/analyze_factorial.py --seed_dirs $S --out $O/factorial.csv
python3 src/analysis/analyze_factorial.py --protected --seed_dirs $S --out $O/factorial_protected.csv
python3 src/analysis/analyze_degraded.py --seed_dirs $S --out $O/degraded.csv
python3 src/analysis/analyze_degraded.py --protected --seed_dirs $S --out $O/degraded_protected.csv
echo "analysis written to $O"
