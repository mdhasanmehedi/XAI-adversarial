#!/bin/bash
# Cross-seed analysis of the functional-validity (v5) re-run. Run from the repository root,
# after run_v5_mac.sh has finished:   bash analyze_v5.sh
# each step reports FAILED instead of stopping the script
R=${1:-results_v5}
S="$R/tables/seed0 $R/tables/seed1 $R/tables/seed2 $R/tables/seed3 $R/tables/seed4"
OUT=$R/tables/cross_seed; mkdir -p $OUT
for s in 0 1 2 3 4; do
  d=$R/tables/seed$s
  # faithfulness is a property of the clean model, which is unchanged -> reuse
  cp results/tables/seed$s/faithfulness_by_ranker*.csv $d/
  # the initial IG run is not repeated: the IG re-run in variants/ is the IG attack
  python3 - "$d" <<'EOF'
import sys, pandas as pd, pathlib
d = pathlib.Path(sys.argv[1])
for sfx in ("", "_protected"):
    parts = sorted((d / f"variants{sfx}").glob("*.csv"))
    if parts:
        pd.concat([pd.read_csv(p) for p in parts]).to_csv(d / f"guided_variant_results{sfx}.csv", index=False)
ig = [pd.read_csv(p) for p in sorted((d / "variants").glob("ig_*.csv"))]
if ig:
    pd.concat(ig).drop(columns=["ranker"]).to_csv(d / "guided_attack_results.csv", index=False)
EOF
done
python3 src/analysis/ranking_control_comparison.py --seed_dirs $S --out $OUT/ranking_control.csv > $OUT/ranking_control.txt || echo "FAILED: python3 src/analysis/ranking_control_comparison.py --seed_dirs $S --out $OUT/ranking_control.csv > $OUT/ranking_control.txt"
python3 src/analysis/protected_prefix_comparison.py --seed_dirs $S --out $OUT/protected_prefix.csv > $OUT/protected_prefix.txt || echo "FAILED: python3 src/analysis/protected_prefix_comparison.py --seed_dirs $S --out $OUT/protected_prefix.csv > $OUT/protected_prefix.txt"
python3 src/analysis/protected_vs_unguided.py --seed_dirs $S --out $OUT/protected_vs_unguided.csv > $OUT/protected_vs_unguided.txt || echo "FAILED: python3 src/analysis/protected_vs_unguided.py --seed_dirs $S --out $OUT/protected_vs_unguided.csv > $OUT/protected_vs_unguided.txt"
python3 src/analysis/analyze_hotflip.py --seed_dirs $S --out $OUT/hotflip_vs_rankers.csv > $OUT/hotflip.txt || echo "FAILED: python3 src/analysis/analyze_hotflip.py --seed_dirs $S --out $OUT/hotflip_vs_rankers.csv > $OUT/hotflip.txt"
python3 src/analysis/analyze_hotflip.py --protected --seed_dirs $S --out $OUT/hotflip_vs_rankers_protected.csv > $OUT/hotflip_protected.txt || echo "FAILED: python3 src/analysis/analyze_hotflip.py --protected --seed_dirs $S --out $OUT/hotflip_vs_rankers_protected.csv > $OUT/hotflip_protected.txt"
python3 src/analysis/faithfulness_vs_efficiency.py --seed_dirs $S --out $OUT/faithfulness_vs_efficiency.csv > $OUT/faithfulness_vs_efficiency.txt || echo "FAILED: python3 src/analysis/faithfulness_vs_efficiency.py --seed_dirs $S --out $OUT/faithfulness_vs_efficiency.csv > $OUT/faithfulness_vs_efficiency.txt"
python3 src/analysis/faithfulness_vs_efficiency.py --protected --seed_dirs $S --out $OUT/faithfulness_vs_efficiency_protected.csv > $OUT/faithfulness_vs_efficiency_protected.txt || echo "FAILED: python3 src/analysis/faithfulness_vs_efficiency.py --protected --seed_dirs $S --out $OUT/faithfulness_vs_efficiency_protected.csv > $OUT/faithfulness_vs_efficiency_protected.txt"
(cd src && python3 -m analysis.pooled_mcnemar --results_base ../$R/tables --out ../$OUT/pooled_mcnemar.csv > ../$OUT/pooled_mcnemar.txt) || echo "FAILED: pooled_mcnemar"
python3 src/analysis/class_prefix_breakdown.py --results_base $R/tables --out $OUT/class_prefix.csv > $OUT/class_prefix.txt || echo "FAILED: class_prefix_breakdown"
echo "analysis written to $OUT"
