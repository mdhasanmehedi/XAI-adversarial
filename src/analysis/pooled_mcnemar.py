"""Pooled across-seed exact McNemar tests (Supplementary Table S26).

For each comparison, the discordant pairs of the five seeds (500 URLs per seed)
are summed and tested with one exact (binomial) McNemar test. V0 comparator:
each seed's reference baseline (the unguided method with the highest targeted
ASR at 10 edits on the seed's 10,000-URL sample), evaluated on the 500 URLs.
V1 comparator: the best protected unguided baseline on the 500 URLs.

Usage (from src/):  python -m analysis.pooled_mcnemar --results_base ../results/tables
"""
import argparse, os
import pandas as pd
from scipy.stats import binomtest

UNGUIDED = ["random_substitution", "random_insertion", "random_deletion",
            "adjacent_swap", "segment_manipulation", "domain_path_tld_transform"]

def success(df, key="url"):
    df = df.drop_duplicates(key)
    s = df["targeted_success"].astype(str).str.lower().eq("true") & (
        pd.to_numeric(df["targeted_edits"], errors="coerce") <= 10)
    return pd.Series(s.values, index=df[key].values)

def load_seed(base, seed):
    d = os.path.join(base, f"seed{seed}")
    out = {}
    for setting, sub in (("V0", "variants"), ("V1", "variants_protected")):
        for f in sorted(os.listdir(os.path.join(d, sub))):
            ranker, mode = f[:-4].split("_", 1)
            out[(setting, mode, ranker)] = success(pd.read_csv(os.path.join(d, sub, f)))
    urls = out[("V0", "one_shot", "ig")].index
    ung = pd.read_csv(os.path.join(d, "unguided_attack_results.csv"))
    best = max(UNGUIDED, key=lambda m: success(ung[ung.method == m]).mean())
    ref = success(ung[ung.method == best]).reindex(urls)
    out[("V0", "ref")] = ref
    prot = {m: success(pd.read_csv(os.path.join(d, "unguided_protected", m + ".csv"))).reindex(urls)
            for m in UNGUIDED}
    out[("V1", "ref")] = prot[max(UNGUIDED, key=lambda m: prot[m].mean())]
    # initial IG runs (Table 3) for the V0 comparison with the reference baseline
    g = pd.read_csv(os.path.join(d, "guided_attack_results.csv"))
    for mode in ("one_shot", "adaptive"):
        col = "mode" if "mode" in g.columns else "method"
        out[("V0", mode, "ig_initial")] = success(g[g[col].astype(str).str.replace("-", "_") == mode]).reindex(urls)
    return out, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", default="../results/tables")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [load_seed(a.results_base, s) for s in a.seeds]
    for s, (_, best) in zip(a.seeds, seeds):
        print(f"seed {s}: V0 reference baseline = {best}")
    rows = []
    def add(setting, mode, mkey, ckey, mname, cname):
        b = c = 0
        for out, _ in seeds:
            m = out[mkey].reindex(out[("V0", "one_shot", "ig")].index).fillna(False)
            r = out[ckey].reindex(m.index).fillna(False)
            b += int((m & ~r).sum()); c += int((~m & r).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append(dict(setting=setting, guidance=mode, method=mname, comparator=cname,
                         method_only=b, comparator_only=c, p_exact=p))
    for setting in ("V0", "V1"):
        refname = "Reference baseline" if setting == "V0" else "Protected reference baseline"
        for mode in ("one_shot", "adaptive"):
            add(setting, mode, (setting, mode, "ig"), (setting, mode, "random"), "ig", "random")
            add(setting, mode, (setting, mode, "ig"), (setting, "ref"), "ig", refname)
            if setting == "V0":
                add(setting, mode, (setting, mode, "ig_initial"), (setting, "ref"), "ig_initial", refname)
            for r in ("occlusion", "saliency", "gradxinput", "random"):
                add(setting, mode, (setting, mode, r), (setting, "ref"), r, refname)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)

if __name__ == "__main__":
    main()
