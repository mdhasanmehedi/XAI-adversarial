"""
Analyse the position x character factorial (attacks/factorial.py).

Run from the repository root (NOT src/):
    python src/analysis/analyze_factorial.py --seed_dirs results/tables/seed0 results/tables/seed1 \
        results/tables/seed2 results/tables/seed3 results/tables/seed4 \
        --out results/tables/cross_seed/factorial.csv
    python src/analysis/analyze_factorial.py --protected --seed_dirs ... \
        --out results/tables/cross_seed/factorial_protected.csv

Prints, per guidance mode:
  (1) mean +- SD targeted ASR at the budget for every position x character cell;
  (2) the CHARACTER effect: grad minus random character, within each position rule
      (paired exact McNemar per seed, BH-corrected; counts of seeds significantly higher/lower);
  (3) the POSITION effect: each position rule minus random positions, within each character rule;
  (4) gains over the random x random cell and the interaction.
Also checks that the (hotflip, grad) cell reproduces <seed_dir>/hotflip[_protected]/hotflip_<mode>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

POS = ["random", "ig", "occlusion", "hotflip"]
CH = ["random", "grad"]


def mcnemar(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return n10, n01, p


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[o] = np.minimum(r, 1.0)
    return out


def load(path, budget):
    d = pd.read_csv(path)
    d["hit"] = d["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)
    return d.set_index("url")["hit"].groupby(level=0).first()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protected", action="store_true")
    ap.add_argument("--budget", type=int, default=10)
    args = ap.parse_args()
    sub = "factorial_protected" if args.protected else "factorial"
    hsub = "hotflip_protected" if args.protected else "hotflip"

    asr, tests = [], []
    for sd in args.seed_dirs:
        sd = Path(sd)
        for mode in ["one_shot", "adaptive"]:
            cells = {}
            for p in POS:
                for c in CH:
                    f = sd / sub / f"{p}_{c}_{mode}.csv"
                    if f.exists():
                        cells[(p, c)] = load(f, args.budget)
            for (p, c), s in cells.items():
                asr.append({"seed": sd.name, "mode": mode, "position": p, "char": c,
                            "asr": float(s.mean()), "n": len(s)})
            hf = sd / hsub / f"hotflip_{mode}.csv"
            if hf.exists() and ("hotflip", "grad") in cells:
                ref = load(hf, args.budget)
                same = (ref.sort_index().values == cells[("hotflip", "grad")].sort_index().values).all()
                print(f"[check] {sd.name}/{mode}: (hotflip,grad) cell reproduces hotflip.py: {same}")
            for p in POS:   # character effect
                if (p, "grad") in cells and (p, "random") in cells:
                    a, b = cells[(p, "grad")], cells[(p, "random")]
                    i = a.index.intersection(b.index)
                    n10, n01, pv = mcnemar(a.loc[i].values, b.loc[i].values)
                    tests.append({"seed": sd.name, "mode": mode, "effect": "character (grad - random)",
                                  "within": f"position={p}", "a_only": n10, "b_only": n01, "p": pv,
                                  "diff": float(a.loc[i].mean() - b.loc[i].mean())})
            for c in CH:    # position effect
                for p in POS[1:]:
                    if (p, c) in cells and ("random", c) in cells:
                        a, b = cells[(p, c)], cells[("random", c)]
                        i = a.index.intersection(b.index)
                        n10, n01, pv = mcnemar(a.loc[i].values, b.loc[i].values)
                        tests.append({"seed": sd.name, "mode": mode,
                                      "effect": f"position ({p} - random)", "within": f"char={c}",
                                      "a_only": n10, "b_only": n01, "p": pv,
                                      "diff": float(a.loc[i].mean() - b.loc[i].mean())})
    A = pd.DataFrame(asr)
    T = pd.DataFrame(tests)
    if A.empty:
        raise SystemExit("no factorial results found")
    T["p_BH"] = np.nan
    for m in T["mode"].unique():
        idx = T.index[T["mode"] == m]
        T.loc[idx, "p_BH"] = bh(T.loc[idx, "p"].values)
    T["sig_up"] = (T.a_only > T.b_only) & (T.p_BH < .05)
    T["sig_down"] = (T.a_only < T.b_only) & (T.p_BH < .05)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    A.to_csv(args.out, index=False)
    T.round(5).to_csv(str(args.out).replace(".csv", "_tests.csv"), index=False)

    for mode in ["one_shot", "adaptive"]:
        a = A[A["mode"] == mode]
        print(f"\n===== {mode}: mean ± SD of targeted ASR@{args.budget} over {a.seed.nunique()} seeds =====")
        g = a.groupby(["position", "char"])["asr"].agg(["mean", "std"])
        tab = g.apply(lambda r: f"{r['mean']:.3f}±{r['std']:.3f}", axis=1).unstack("char")
        print(tab.reindex(POS).to_string())
        base = a[(a.position == "random") & (a.char == "random")]["asr"].mean()
        gain = a.groupby(["position", "char"])["asr"].mean() - base
        print("\nGain over random x random (mean ASR difference):")
        print(gain.unstack("char").reindex(POS).round(3).to_string())
        print("\nSeeds with a significant effect (BH, alpha .05): up/down, mean ASR difference")
        t = T[T["mode"] == mode].groupby(["effect", "within"]).agg(
            up=("sig_up", "sum"), down=("sig_down", "sum"), mean_diff=("diff", "mean")).round(3)
        print(t.to_string())
    print(f"\nWritten to {args.out} and {str(args.out).replace('.csv', '_tests.csv')}")


if __name__ == "__main__":
    main()
