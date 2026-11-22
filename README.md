# XAI-adversarial

Code and result files for:

> **Does Attribution Help Evade Malicious URL Detectors? Controlled Ranking, Dataset Shortcuts, and Faithfulness Across Seeds**
> Md Mehedi Hasan — Independent researcher, Bogura, Bangladesh
> Corresponding author: hasanmd24@outlook.com

---

## Overview

The study asks whether attribution (Integrated Gradients and three other ranking methods) makes evasion of a character-level malicious-URL detector more successful, whether the advantage depends on a dataset shortcut (the URL prefix), and whether independently measured attribution faithfulness is associated with it. The main corpus is attacked with **five independently trained target models**; the core experiment is replicated on two further corpora (PhiUSIIL, GramBeddings) and on a second architecture (BiLSTM), with three seeds each.

* **Main corpus:** Malicious URLs dataset (Kaggle, `sid321axn`, 2021; CC0), four classes (benign, phishing, malware, defacement). Not redistributed here; place it at `data/raw/malicious_phish.csv`.
* **Replication corpora:** PhiUSIIL (UCI Machine Learning Repository, id 967, CC BY 4.0) and GramBeddings (distributed by its authors). Not redistributed here.
* **Detectors:** character CNN-Transformer (658,629 parameters) and character BiLSTM (601,413 parameters), both in `src/model.py`.

## Two validity predicates and three result folders

Every candidate perturbation must pass a validity predicate (`src/attacks/validity.py`, paper §3.3):

* **syntactic** (default): parseable URL, well-formed percent-encoding, non-empty host without spaces or control characters, alphabetic final host label;
* **functional** (`URL_VALIDITY_MODE=functional`): additionally an http, https or empty scheme and a host made of DNS letter-digit-hyphen labels under a real public suffix (or an IPv4 address), checked offline with `tldextract`.

| Folder | Predicate | Contents |
|---|---|---|
| `results_v5/tables/` | functional | Main corpus, 5 seeds: unguided baselines (10,000 URLs per seed), Integrated Gradients and the other rankers (V0 and V1), random-ranking control, protected unguided baselines, adapted HotFlip; `cross_seed/` summaries |
| `results_v6/gram/`, `results_v6/bilstm/` | functional | GramBeddings and BiLSTM replications, 3 seeds, 500 URLs per seed |
| `results_v6/paper_tables/` | functional | Every functional-validity table of the paper as Markdown, written by `src/analysis/paper_tables.py` |
| `results/tables/` | syntactic | Main corpus, earlier run: within-method faithfulness analysis (H6), surrogate transfer, defenses, position-by-character factorial, graded degradation; also the syntactic versions of the attacks used in Table S40 |
| `results/phiusiil/` | syntactic | PhiUSIIL replication |
| `results/gram/`, `results/bilstm/` | syntactic | Earlier versions of the two replications (superseded by `results_v6/`, kept for comparison) |
| `runs/*/table1_results.csv` | — | Clean performance of the target models, the BiLSTM and the surrogate |

Faithfulness (`faithfulness_by_ranker*.csv`) is computed on the clean models and does not depend on the predicate; the files in `results_v5/` and `results_v6/` are copies of those in `results/`.

## Where each table and figure comes from

Paper numbering (main text Tables 1–9 and Figures 1–7; Supplementary Material Tables S1–S40 and Figures S1–S5).

| Paper item | Source files | Script |
|---|---|---|
| Table 3, S1, S2, S28 (guided vs. unguided; budget sweep) | `results_v5/tables/seedN/unguided_attack_results.csv`, `variants/ig_*.csv` | `src/analysis/paper_tables.py` |
| Table 4, S8 (ranking control, V0) | `results_v5/tables/seedN/variants/` | `paper_tables.py`; also `ranking_control_comparison.py` |
| Table 5, S9, S9b (prefix frozen, V1) | `variants_protected/`, `unguided_protected/` | `paper_tables.py`; also `protected_prefix_comparison.py`, `protected_vs_unguided.py` |
| Table 6, S10a, S10b (faithfulness of each ranking) | `seedN/faithfulness_by_ranker*.csv` | `faithfulness_by_ranker.py` |
| Table 7, S12, S22 (faithfulness vs. success) | `variants*/`, `faithfulness_by_ranker*.csv` | `paper_tables.py`; also `faithfulness_vs_efficiency.py` |
| Table 8, S15 (adapted HotFlip) | `results_v5/tables/seedN/hotflip*/` | `paper_tables.py`; also `analyze_hotflip.py` |
| Table 9 (hypothesis summary) | the tables above | — |
| Table S7 (host-validity sensitivity) | `results_v5/tables/` | `paper_tables.py` |
| Table S11, S39 (prefix diagnostic, per-class ASR) | `results_v5/tables/` | `paper_tables.py`; also `class_prefix_breakdown.py` |
| Table S26 (pooled McNemar) | `results_v5/tables/` | `paper_tables.py`; also `pooled_mcnemar.py` |
| Table S40 (syntactic vs. functional) | `results/tables/` and `results_v5/tables/` | `paper_tables.py` |
| Tables S14, S34 (GramBeddings) | `results_v6/gram/` | `paper_tables.py` |
| Tables S19, S20, S20b, S35 (BiLSTM) | `results_v6/bilstm/` | `paper_tables.py` |
| Tables S13, S23, S33 (PhiUSIIL) | `results/phiusiil/` | `analyze_phiusiil.sh` |
| Tables S21, S24, S25 (replication faithfulness) | `results/bilstm/`, `results/gram/`, `results/phiusiil/` (`faithfulness_by_ranker*.csv`) | `faithfulness_by_ranker.py` |
| Tables S3, S29, Figure S1 (within-method analysis, H6) | `results/tables/seedN/faithfulness_results.csv`, `seedN_table4_*` | `faithfulness/correlate_with_attacks.py` |
| Table S27 (clean performance) | `runs/runs_v1/table1_results.csv` | `train.py` |
| Table S30, Figure S3 (surrogate transfer) | `results/tables/seedN/seedN_table5_results.csv`, `runs/surrogate/` | `attacks/transfer_evaluate.py` |
| Tables S4, S5, S31, S32, Figure S4 (defenses) | `seedN_table6_*`, `seedN_table7_results.csv`, `table6_attack_battery_*` | `defense/` |
| Tables S16, S36, S37 (factorial) | `results/tables/seedN/factorial*/`, `cross_seed/factorial*` | `analyze_factorial.py` |
| Tables S17, S18, S38, Figure S5 (graded degradation) | `results/tables/seedN/degraded*/`, `cross_seed/degraded*` | `analyze_degraded.py` |

To regenerate every functional-validity table (from the repository root):

```bash
python src/analysis/paper_tables.py --main results_v5/tables --main_syntactic results/tables \
    --gram results_v6/gram --bilstm results_v6/bilstm --outdir results_v6/paper_tables
```

## Repository structure

```
XAI-adversarial/
├── README.md, LICENSE, requirements.txt, .gitignore
├── run_v5_mac.sh, analyze_v5.sh     # functional-validity re-run of the main-corpus attacks
├── run_v6_mac.sh                    # functional-validity re-run of the GramBeddings and BiLSTM replications
├── apply_bilstm_patch.py, analyze_bilstm.sh   # BiLSTM option (already applied to src/model.py and src/train.py)
├── src/
│   ├── model.py, train.py, data_prep.py, aggregate_cross_seed.py
│   ├── prep_phiusiil.py, prep_grambeddings.py, run_*_pipeline.sh, analyze_*.sh
│   ├── attacks/      # transforms, validity (syntactic + functional), attribution, unguided,
│   │                 # unguided_protected, guided, guided_variants, hotflip, factorial,
│   │                 # degraded_attribution, transfer_evaluate, build_eval_sample
│   ├── faithfulness/ # deletion_auc, faithfulness_by_ranker, correlate_with_attacks
│   ├── analysis/     # paper_tables (all functional-validity tables), ranking_control_comparison,
│   │                 # protected_prefix_comparison, protected_vs_unguided, faithfulness_vs_efficiency,
│   │                 # analyze_hotflip, analyze_factorial, analyze_degraded, pooled_mcnemar,
│   │                 # class_prefix_breakdown, host_validity_sensitivity
│   └── defense/      # train_defended, evaluate_defense
├── results/          # syntactic-predicate results (see table above)
├── results_v5/       # functional-predicate main-corpus results
├── results_v6/       # functional-predicate replications and paper_tables/
└── runs/             # clean-performance summaries (checkpoints are not included)
```

## Reproducing

Experiments were run on Apple Silicon (MPS backend) with PyTorch ≥ 2.0. MPS kernels are not bit-for-bit deterministic, so retraining with the same seeds gives slightly different models; every table can be reproduced exactly from the per-URL result files in this repository. The trained checkpoints are not included (available from the author on request).

```bash
pip install -r requirements.txt
cd src
python data_prep.py --input ../data/raw/malicious_phish.csv --outdir ../data/processed/split_v1 --seed 0
python train.py --split_dir ../data/processed/split_v1 --outdir ../runs/runs_v1 --seeds 0 1 2 3 4 --epochs 10
bash run_seed_pipeline.sh N        # per seed: evaluation sample, attacks, faithfulness (syntactic predicate)
cd ..
caffeinate -i bash run_v5_mac.sh    # main-corpus attacks under the functional predicate -> results_v5/
bash analyze_v5.sh
caffeinate -i bash run_v6_mac.sh    # GramBeddings and BiLSTM replications under the functional predicate -> results_v6/
python src/analysis/paper_tables.py # all functional-validity tables -> results_v6/paper_tables/
```

On Linux, drop `caffeinate -i`. The replication corpora are prepared with `src/prep_phiusiil.py` and `src/prep_grambeddings.py` and trained with `train.py` (`--arch_type bilstm` for the BiLSTM).

## Methodology notes

* **Query accounting.** One applied edit is one classifier query. Ranking computations (Integrated Gradients, occlusion's forward passes, HotFlip's gradient) are white-box and not counted as queries.
* **Statistics.** Exact McNemar tests on the same URLs with Benjamini–Hochberg correction; families as stated in §4.7 of the paper and in each table caption.
* **Faithfulness** is measured on the clean classifier and never from attack outcomes.

## Ethics and dual use

All experiments were run offline against locally trained models and a static public dataset. No live phishing, malware or defacement infrastructure was contacted, registered or interacted with. The validity predicates check that a URL could be requested, not that it resolves or remains malicious.

## Citation and license

Citation details will be added upon publication. Released under the MIT License (see `LICENSE`). The datasets are distributed by their authors under their own licenses and are not covered by this license.
