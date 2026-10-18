# XAI-adversarial

Code and result files for:

> **Does Attribution Help Evade Malicious URL Detectors? Controlled Ranking, Dataset Shortcuts, and Faithfulness Across Seeds**
> Md Mehedi Hasan — Independent researcher, Bogura, Bangladesh
> Corresponding author: hasanmd24@outlook.com
> https://github.com/mdhasanmehedi/XAI-adversarial

---

## Overview

This study asks whether attribution (Integrated Gradients and other ranking methods) makes evasion of a malicious-URL detector more *efficient*, whether that efficiency depends on a dataset shortcut, and whether independently measured attribution faithfulness is associated with it. Every experiment is run across **five independently trained target-model seeds** on the main corpus (same data split; seeds differ in initialization and data order), with additional three-seed checks on two further corpora and a second detector architecture.

**Dataset:** Malicious URLs dataset (Kaggle, `sid321axn`, 2021; CC0 public domain), four classes (benign, phishing, malware, defacement). After deduplication and near-duplicate grouping, split 70/10/20 into train/validation/test.

**Target model:** Character-level CNN-Transformer, 658,629 parameters.

**Second architecture:** Character-level BiLSTM, 601,413 parameters (three-seed replication, §4.5/§10.14 of the manuscript).

**Replication corpora:** PhiUSIIL (UCI Machine Learning Repository, id 967, CC BY 4.0) and GramBeddings (distributed by its authors). Neither dataset is redistributed here.

---

## What the study does

1. **Central comparison (Tables 6–7):** one-shot and adaptive Integrated-Gradients-guided attacks vs. unguided baselines, at a budget of 10 edits, under a syntactic-validity predicate.
2. **Ranking control (Table 8):** a random-ranking control using the *same* guided action space, plus three alternative attribution rankers (occlusion, saliency, gradient × input), to isolate whether the ranking itself — not just the action space — drives the advantage.
3. **Prefix-frozen setting, V1 (Tables 9–11):** the scheme and a leading `www.` are frozen to test whether the advantage survives once an obvious URL-structure shortcut is removed.
4. **Faithfulness analysis (Tables 12–14):** Deletion AUC, comprehensiveness and sufficiency, measured independently on the clean classifier, correlated with attack efficiency at the method level, the URL level, and within a method (class-adjusted).
5. **Surrogate transfer and defense (Tables 15–17):** targeted transfer to independently trained targets; adversarial-training and character-augmentation defenses evaluated against the attacks.
6. **Replications (Tables 18–21, 26–27):** the same core comparisons re-run on PhiUSIIL, GramBeddings, and a BiLSTM detector.
7. **External benchmark (Table 22):** an adapted HotFlip attack (Ebrahimi et al., 2018) compared against the attribution rankers.
8. **Mechanism analyses (Tables 23–25):** a position-by-character factorial, and a graded degradation of the Integrated Gradients ranking toward random order.

See the manuscript's Table 28 for the outcome of every hypothesis across all four settings (main corpus, PhiUSIIL, GramBeddings, BiLSTM).

---

## Repository structure

```
XAI-adversarial/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore                      # checkpoints, raw/processed data, caches
├── apply_bilstm_patch.py           # adds the BiLSTM option to model.py, train.py, attacks/unguided.py
├── analyze_bilstm.sh                # BiLSTM cross-seed analysis (Tables 26-27, S19-S21)
│
├── src/
│   ├── model.py                    # CharCNNTransformer (658,629 params); CharBiLSTM (601,413)
│   ├── train.py                    # AdamW, best-validation-macro-F1 checkpoint; --arch_type
│   ├── data_prep.py                # dedup, near-duplicate grouping, stratified 70/10/20 split
│   ├── aggregate_cross_seed.py     # cross-seed means and SDs
│   ├── run_seed_pipeline.sh        # main-corpus pipeline for one seed
│   ├── run_hotflip.sh              # HotFlip benchmark (Table 22)
│   ├── run_factorial.sh            # position-by-character factorial (Tables 23-24)
│   ├── run_degraded.sh             # graded degradation of the ranking (Table 25)
│   ├── run_bilstm_pipeline.sh      # BiLSTM replication, one seed
│   ├── prep_phiusiil.py            # PhiUSIIL preparation and split
│   ├── run_phiusiil_pipeline.sh
│   ├── analyze_phiusiil.sh         # Tables 18-19
│   ├── prep_grambeddings.py        # GramBeddings pooling, re-split, subsampling
│   ├── run_gram_pipeline.sh
│   ├── analyze_gram.sh             # Tables 20-21
│   │
│   ├── attacks/
│   │   ├── transforms.py           # six unguided transformations
│   │   ├── validity.py             # syntactic-validity predicate
│   │   ├── attribution.py          # Integrated Gradients and the other rankers
│   │   ├── build_eval_sample.py    # frozen evaluation samples
│   │   ├── unguided.py             # unguided baselines (Table 6)
│   │   ├── unguided_protected.py   # unguided baselines, prefix frozen (V1)
│   │   ├── guided.py               # one-shot / adaptive IG-guided attacks (Table 7)
│   │   ├── guided_variants.py      # ranking methods, random control, --protect_prefix
│   │   ├── compare_guided_vs_unguided.py  # McNemar / Wilcoxon / FDR
│   │   ├── transfer_evaluate.py    # surrogate transfer (Table 15)
│   │   ├── hotflip.py              # adapted HotFlip (Table 22)
│   │   ├── factorial.py            # position x character factorial (Tables 23-24)
│   │   └── degraded_attribution.py # degraded IG ranking (Table 25)
│   │
│   ├── faithfulness/
│   │   ├── deletion_auc.py         # Deletion AUC, comprehensiveness, sufficiency
│   │   ├── correlate_with_attacks.py  # within-method correlations (Table 14)
│   │   └── faithfulness_by_ranker.py  # faithfulness of every ranker
│   │
│   ├── analysis/
│   │   ├── ranking_control_comparison.py   # Table 8
│   │   ├── protected_prefix_comparison.py  # Table 10 (vs. IG, random ranking)
│   │   ├── protected_vs_unguided.py        # Table 10 (vs. protected reference baseline)
│   │   ├── faithfulness_vs_efficiency.py   # Tables 12-13
│   │   ├── host_validity_sensitivity.py    # Supplementary Table S7
│   │   ├── analyze_hotflip.py              # Table 22
│   │   ├── analyze_factorial.py            # Tables 23-24
│   │   └── analyze_degraded.py             # Table 25
│   │
│   └── defense/
│       ├── train_defended.py       # FGM / charaug / combined / none
│       └── evaluate_defense.py     # three-attack battery (Tables 16-17)
│
├── results/
│   ├── tables/                     # main corpus, 5 seeds
│   │   ├── cross_seed/             # cross-seed summaries
│   │   └── seed0/ ... seed4/       # per-seed raw results
│   │       # includes: unguided_attack_results.csv, guided_attack_results.csv,
│   │       # faithfulness_results.csv, seedN_tableX_results.csv,
│   │       # variants/, variants_protected/, unguided_protected/,
│   │       # hotflip/, hotflip_protected/, factorial/, factorial_protected/,
│   │       # degraded/, degraded_protected/
│   ├── phiusiil/                   # 3 seeds + cross_seed/
│   ├── gram/                       # 3 seeds + cross_seed/
│   └── bilstm/                     # 3 seeds + cross_seed/
│
└── runs/
    ├── runs_v1/table1_results.csv      # CNN-Transformer clean performance
    ├── runs_bilstm/table1_results.csv  # BiLSTM clean performance
    ├── surrogate/table1_results.csv    # surrogate clean performance
    └── sanity_check/table1_results.csv
```

---

## Where each table of the paper comes from

| Paper table | Source |
|---|---|
| Table 5 (clean performance) | `runs/runs_v1/table1_results.csv` |
| Table 6, S1 (unguided/guided ASR across budgets) | `results/tables/seedN/*table2*`, aggregated in `cross_seed/table2_cross_seed.csv` |
| Table 7, S2 (guided vs. unguided significance tests) | `seedN_table3_results.csv`, `cross_seed/table3_cross_seed.csv` |
| Table 8, S8 (ranking control) | `results/tables/seedN/variants/`, `variants_protected/` |
| Table 9 | seed-0 prefix diagnostic, single seed |
| Table 10, S9 (prefix-frozen) | `unguided_protected/`, `variants_protected/` |
| Table 11, S10 (faithfulness by ranker) | `faithfulness_by_ranker*.csv` |
| Tables 12-14, S3, S11-S12 (faithfulness-efficiency correlation) | `seedN_table4_results.csv`, `faithfulness_vs_efficiency*` |
| Table 15 (surrogate transfer) | `seedN_table5_results.csv`, `cross_seed/table5_cross_seed.csv` |
| Tables 16-17, S4-S5 (defense) | `seedN_table6_results.csv`, `seedN_table7_results.csv`, `table6_attack_battery_*` |
| Tables 18-19, S13 (PhiUSIIL) | `results/phiusiil/` |
| Tables 20-21, S14 (GramBeddings) | `results/gram/` |
| Table 22, S15 (HotFlip) | `*/hotflip*`, `*/cross_seed/hotflip_vs_rankers*.csv` |
| Tables 23-24, S16 (factorial) | `*/factorial*` |
| Table 25, S17-S18 (graded degradation) | `*/degraded*` |
| Tables 26-27, S19-S21 (BiLSTM) | `results/bilstm/` |
| Table 28 (hypothesis summary) | computed from the tables above |
| S7 (host-validity sensitivity) | `src/analysis/host_validity_sensitivity.py` |

---

## Reproducing the pipeline

Experiments were run on Apple Silicon (MPS backend) with PyTorch >= 2.0; dependencies are in `requirements.txt`. Commands run from `src/` unless noted.

```bash
git clone https://github.com/mdhasanmehedi/XAI-adversarial.git
cd XAI-adversarial
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Prepare the split:** `python data_prep.py --input ../data/raw/malicious_phish.csv --outdir ../data/processed/split_v1 --seed 0`
2. **Train target models (5 seeds):** `python train.py --split_dir ../data/processed/split_v1 --outdir ../runs/runs_v1 --seed N --epochs 10`
3. **Freeze the shared evaluation sample:** `python -m attacks.build_eval_sample ...`
4. **Run the main-corpus pipeline for each seed:** `bash run_seed_pipeline.sh N`
5. **Run the ranking-control, prefix-frozen, HotFlip, factorial and degradation analyses:** see `run_hotflip.sh`, `run_factorial.sh`, `run_degraded.sh`, and the `--protect_prefix` flag on `attacks/guided_variants.py` / `attacks/unguided_protected.py`.
6. **Run the replications:** `run_phiusiil_pipeline.sh`, `run_gram_pipeline.sh`, and (after `python apply_bilstm_patch.py` from the repo root) `run_bilstm_pipeline.sh`.
7. **Aggregate across seeds:** `python aggregate_cross_seed.py --results_base ../results/tables --outdir ../results/tables/cross_seed --seeds 0 1 2 3 4`

The dataset is **not included**; download it from Kaggle and place it at `data/raw/malicious_phish.csv`. Model checkpoints, training logs, and processed/raw data are excluded via `.gitignore` and regenerated by steps 1-3.

---

## Methodology notes

**Validity constraint.** Every candidate perturbation is checked against a syntactic-validity predicate (length, percent-encoding, parseable host, TLD-pattern preservation) before being counted as an edit. This is a syntactic check, not an RFC 3986 grammar validator or a functional/resolution check; every ASR figure in this repository reflects syntactic validity only.

**Query accounting.** One accepted, applied edit = one classifier query. Attribution computation (Integrated Gradients and the other rankers) is white-box and not counted as a query.

**FDR correction.** Benjamini-Hochberg correction is applied to paired attack comparisons within seed (main-corpus ranking-control and prefix comparisons) or within setting and guidance mode pooled across all five seeds (HotFlip, factorial, degradation); see §9 of the manuscript for the exact grouping used for each table.

**Faithfulness** is measured on the clean classifier with no attack involved, so the faithfulness-efficiency correlation is non-circular by construction.

---

## Ethics and dual-use

All experiments were conducted offline against locally trained models and a static, publicly available dataset. No live phishing, malware, or defacement infrastructure was targeted, registered, or interacted with. This repository does not release attack code beyond what reproduces the reported experiments against the study's own models, and the attacks apply only syntactic string edits whose functional validity was not tested. See the manuscript's Ethics and Dual-Use Statement for the full discussion.

---

## Citation

Citation details will be added upon publication.

---

## License

Released under the [MIT License](LICENSE). Please cite the paper if you use this code. The datasets used are distributed separately by their respective authors under their own licenses (see "Overview" above) and are not covered by this repository's license.
