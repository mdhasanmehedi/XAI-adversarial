# Attribution-Guided Evasion of Malicious URL Detectors

Code and results for:

> **Attribution-Guided Evasion of Malicious URL Detectors: A Multi-Seed Study of Faithfulness, Transfer, and Defense**  
> Md Mehedi Hasan — Independent researcher, Bogura, Bangladesh  
> Contact: hasanmd24@outlook.com  
> https://github.com/mdhasanmehedi/XAI-adversarial

---

## Overview

This study investigates whether Integrated Gradients attribution can serve as an effective attack resource against a character-level CNN-Transformer malicious URL detector, and whether explanation faithfulness is statistically associated with adversarial attack efficiency. All experiments are replicated across **five independently seeded target models** (same data split; the seeds differ in initialization and data order) — a design choice that materially revised several conclusions relative to what a single trained model would have suggested.

**Dataset:** 651,191-URL four-class benchmark (benign, phishing, malware, defacement; Kaggle "Malicious URLs dataset", CC0 public domain, compiled from ISCX-URL-2016, a malware-domain blacklist, a benign-URL repository, PhishTank and PhishStorm). After exact-duplicate removal: 641,119 unique URLs, split 70/10/20 into train/validation/test.

**Target model:** Character-level CNN-Transformer, 658,629 parameters. Clean accuracy 97.9% ± 0.2%, macro-F1 96.8% ± 0.3% across five seeds.

**Surrogate model:** Architecturally distinct smaller variant (117,541 parameters; clean accuracy 97.87%) used for transfer experiments.

---

## Key Findings (as reported in the paper)

All statements below follow the final manuscript. "ASR" is targeted attack success rate at the maximum budget of 10 edits; the five seeds share one data split.

**H1 — Attribution-guided attacks achieve higher attack success than the evaluated unguided baselines (supported at 10 edits).**
Mean ASR: one-shot 30.7%, adaptive 26.8%, strongest unguided baseline (`random_insertion`) 15.4%. Both guided attacks exceeded each seed's strongest unguided baseline in all five seeds; the difference was statistically significant (exact McNemar test, Benjamini–Hochberg FDR within each seed) for one-shot in four seeds and for adaptive in two. The reference baseline is chosen from the observed results in each seed, so this is a best-baseline comparison, not a pre-registered test. The guided and unguided attacks also differ in edit types and other settings, and no random-ranking control was run, so the result does not isolate attribution as the cause.

**H2 — Lower edit cost is not consistently supported.**
Mean effect sizes are small (+0.03 one-shot, −0.07 adaptive); per-seed values range from −0.43 to +0.68 (one-shot) and −0.63 to +0.39 (adaptive), and fewer edits than the reference occur in only 3 of 5 seeds for each method.

**H3 — The faithfulness–efficiency association is confounded by class; evidence after class adjustment is limited.**
Pooled within-seed Spearman correlations were significantly positive (the direction opposite to H3) in 13 of 40 seed × method combinations; none of these remains significantly positive after class adjustment, and 4 adjusted correlations are significantly negative (the direction predicted by H3). The confidence intervals are unadjusted bootstrap intervals, so these counts are descriptive.

**H4 — Targeted transfer to independently trained targets (supported descriptively).**
All seven methods achieved targeted transfer, with mean rates of 29–48% across five targets. Surrogate attack strength did not predict transfer (Spearman ρ = 0.07, p = 0.88, n = 7 methods): the method with the highest surrogate ASR (`one_shot`, 42.8%) ranks fourth in transfer, whereas `domain_path_tld_transform`, the second-weakest against the surrogate (10.4%), transfers best.

**H5 — The combined defense is the most consistently robust condition (descriptive).**
Mean ASR against the unseen one-shot attack: control 34.6%, FGM 35.6%, character augmentation 19.8%, combined 16.7%. The combined defense was below the control in all five seeds; FGM and character augmentation each in only two. Clean accuracy varied by 0.3 percentage points (97.8–98.1%), so the evaluation provides no evidence of a material clean-accuracy cost. The faithfulness comparison with the control is exploratory (no significance test).

---

## Where each table of the paper comes from

| Paper table | Repository file(s) |
|---|---|
| Table 4 (clean performance) | `runs/runs_v1/table1_results.csv` |
| Table 5 and Table S1 (unguided/guided ASR across budgets) | `results/tables/seedN/seedN_table2_results.csv`, aggregated in `cross_seed/table2_cross_seed.csv` |
| Table 6 and Table S2 (guided vs. reference tests) | `seedN_table3_results.csv`, aggregated in `cross_seed/table3_cross_seed.csv` |
| Table 7 and Table S3 (faithfulness–efficiency correlations) | `seedN_table4_results.csv` (also `seedN_table4_by_class.csv`), summary in `cross_seed/table4_replication_summary.csv`; seed-level correlation in `cross_seed/table4_seed_level.csv` |
| Table 8 (surrogate transfer) | `seedN_table5_results.csv`, aggregated in `cross_seed/table5_cross_seed.csv` |
| Table 9 and Table S4 (defense ASR) | `seedN_table6_results.csv` and `table6_attack_battery_*` (three-attack battery) |
| Table 10 and Table S5 (defense faithfulness) | `seedN_table7_results.csv` |
| Table S7 (host-validity sensitivity) | `src/analysis/host_validity_sensitivity.py` |

---

## Repository Structure

```
XAI-adversarial/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── model.py                      # CharCNNTransformer (658,629 params)
│   ├── train.py                      # Training loop, AdamW, best-val-F1 checkpointing
│   ├── data_prep.py                  # Dedup + near-dup grouping + stratified split
│   ├── aggregate_cross_seed.py       # Aggregates per-seed CSVs into cross-seed summaries
│   │
│   ├── analysis/
│   │   └── host_validity_sensitivity.py  # Host-validity sensitivity analysis (paper Table S7)
│   │
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── transforms.py             # Six unguided transformations (four character-level, two structural)
│   │   ├── validity.py               # Syntactic-validity predicate (urlsplit-based; not a strict RFC 3986 validator)
│   │   ├── attribution.py            # Integrated Gradients
│   │   ├── build_eval_sample.py      # Freezes the shared 10,000-URL evaluation sample
│   │   ├── unguided.py               # Six unguided baseline attack families
│   │   ├── guided.py                 # One-shot and adaptive IG-guided attacks
│   │   ├── transfer_evaluate.py      # Surrogate-to-target transfer evaluation (paper Table 8)
│   │   └── compare_guided_vs_unguided.py  # McNemar + Wilcoxon + FDR (paper Table 6)
│   │
│   ├── faithfulness/
│   │   ├── __init__.py
│   │   ├── deletion_auc.py           # Deletion AUC + Comprehensiveness/Sufficiency
│   │   └── correlate_with_attacks.py # Spearman + bootstrap CI + class adjustment (paper Table 7)
│   │
│   └── defense/
│       ├── __init__.py
│       ├── train_defended.py         # FGM / charaug / combined / none conditions
│       └── evaluate_defense.py       # Three-attack generalization battery (paper Tables 9–10)
│
├── results/
│   └── tables/
│       ├── cross_seed/                     # Aggregated cross-seed summaries
│       ├── seed0/                          # Per-seed results (15 files)
│       ├── seed1/                          # Per-seed results (15 files)
│       ├── seed2/                          # Per-seed results (15 files)
│       ├── seed3/                          # Per-seed results (15 files)
│       └── seed4/                          # Per-seed results (15 files)
│
└── runs/
    ├── runs_v1/table1_results.csv          # Target model clean performance
    ├── surrogate/table1_results.csv        # Surrogate model clean performance
    └── sanity_check/table1_results.csv     # Split verification run
```

Each `seedN/` folder contains 15 files. The per-URL files (`faithfulness_results.csv`, `guided_attack_results.csv`, `unguided_attack_results.csv`) are unprefixed. The four battery files are named `table6_attack_battery_<defense>_seed0.csv` in `seed0/` and `seedN_table6_attack_battery_<defense>.csv` in `seed1/`–`seed4/`:

| File | Contents |
|---|---|
| `faithfulness_results.csv` | Per-URL Deletion AUC, Comprehensiveness, Sufficiency |
| `guided_attack_results.csv` | Per-URL one-shot and adaptive attack outcomes |
| `unguided_attack_results.csv` | Per-URL outcomes for all six unguided methods |
| `seedN_table2_results.csv` | ASR and edit cost across budgets (paper Table 5, Table S1) |
| `seedN_table3_results.csv` | Guided vs. unguided significance test results (paper Table 6, Table S2) |
| `seedN_table3_guided_only.csv` | Guided-method budget sweep summary |
| `seedN_table4_by_class.csv` | Faithfulness–efficiency correlations by URL class |
| `seedN_table4_results.csv` | Faithfulness–efficiency correlations overall (paper Table 7, Table S3) |
| `seedN_table5_results.csv` | Surrogate transfer results (paper Table 8) |
| `seedN_table6_results.csv` | Defense evaluation — ASR and robust accuracy (paper Table 9, Table S4) |
| `seedN_table7_results.csv` | Defense evaluation — faithfulness change (paper Table 10, Table S5) |
| `…table6_attack_battery_charaug….csv` | Three-attack battery results (seen-type, unseen-structural, unseen-guided) for the charaug-defended model |
| `…table6_attack_battery_combined….csv` | Same battery, combined defense |
| `…table6_attack_battery_fgm….csv` | Same battery, FGM |
| `…table6_attack_battery_none….csv` | Same battery, retrained undefended control |

---

## Dataset

The study uses the publicly available Kaggle "Malicious URLs dataset" (`malicious_phish.csv`; https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset; license CC0 public domain). The raw file is **not included** here — download it from Kaggle.

Place it at `data/raw/malicious_phish.csv` before running `data_prep.py`.

---

## Setup

```bash
git clone https://github.com/mdhasanmehedi/XAI-adversarial.git
cd XAI-adversarial
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Experiments were run on Apple Silicon (MPS backend) with PyTorch ≥ 2.0 (see `requirements.txt`). CUDA and CPU are also supported — `train.py` auto-detects the available device.

---

## Reproducing the Pipeline

All commands run from `src/` with the virtual environment active.

### 1. Prepare the data split
```bash
python data_prep.py \
    --input ../data/raw/malicious_phish.csv \
    --outdir ../data/processed/split_v1 \
    --seed 0
```

### 2. Train target models (five seeds)
```bash
for SEED in 0 1 2 3 4; do
    python train.py \
        --split_dir ../data/processed/split_v1 \
        --outdir ../runs/runs_v1 \
        --seed $SEED --epochs 10
done
```

### 3. Train surrogate model
```bash
python train.py \
    --split_dir ../data/processed/split_v1 \
    --outdir ../runs/surrogate \
    --seed 100 --epochs 10 \
    --embed_dim 32 --conv_channels 64 \
    --transformer_layers 1 --transformer_heads 2
```

### 4. Freeze the shared evaluation sample
This step is critical — every attack method must run on the same URLs for McNemar's paired tests to be valid.
```bash
python -m attacks.build_eval_sample \
    --split_dir ../data/processed/split_v1 \
    --checkpoint ../runs/runs_v1/model_seed0.pt \
    --outdir ../data/processed \
    --n 10000 --seed 0
```

### 5. Run attacks (for each seed N)
```bash
# Unguided baselines (n=10,000 URLs, all 6 methods)
python -m attacks.unguided \
    --split_dir ../data/processed/split_v1 \
    --checkpoint ../runs/runs_v1/model_seedN.pt \
    --eval_sample ../data/processed/attack_eval_sample_v1.csv \
    --outdir ../results/tables/seedN \
    --max_budget 10 --seed N

# Attribution-guided attacks (n=500 URLs, IG computation)
python -m attacks.guided \
    --split_dir ../data/processed/split_v1 \
    --checkpoint ../runs/runs_v1/model_seedN.pt \
    --eval_sample ../data/processed/attack_eval_sample_v1.csv \
    --outdir ../results/tables/seedN \
    --modes one_shot adaptive --max_budget 10 --seed N
```

### 6. Faithfulness evaluation (for each seed N)
```bash
python -m faithfulness.deletion_auc \
    --split_dir ../data/processed/split_v1 \
    --checkpoint ../runs/runs_v1/model_seedN.pt \
    --eval_sample ../data/processed/attack_eval_sample_v1.csv \
    --outdir ../results/tables/seedN \
    --ig_steps 20 --topk_fraction 0.2 \
    --n_urls 1000 --seed N
```

### 7. Surrogate transfer (for each seed N)
```bash
python -m attacks.transfer_evaluate \
    --target_checkpoint ../runs/runs_v1/model_seedN.pt \
    --surrogate_results_dir ../results/tables/surrogate \
    --outdir ../results/tables/seedN
```

### 8. Defense training and evaluation (for each seed N, each defense)
```bash
# Train defended models
for DEFENSE in none fgm charaug combined; do
    python -m defense.train_defended \
        --split_dir ../data/processed/split_v1 \
        --outdir ../runs/defense \
        --seed N --epochs 10 --defense $DEFENSE
done

# Evaluate against the three-attack generalization battery
for DEFENSE in none fgm charaug combined; do
    python -m defense.evaluate_defense \
        --split_dir ../data/processed/split_v1 \
        --defense_checkpoint ../runs/defense/${DEFENSE}_seedN.pt \
        --undefended_checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables/seedN \
        --max_budget 10 --n_urls 500 --seed N
done
```

### 9. Statistical analysis (for each seed N)
```bash
python -m attacks.compare_guided_vs_unguided --outdir ../results/tables/seedN
python -m faithfulness.correlate_with_attacks --outdir ../results/tables/seedN
```

### 10. Aggregate across seeds
```bash
python aggregate_cross_seed.py \
    --results_base ../results/tables \
    --outdir ../results/tables/cross_seed \
    --seeds 0 1 2 3 4
```

### 11. Host-validity sensitivity analysis (paper Table S7)
```bash
python analysis/host_validity_sensitivity.py \
    --seed_dirs ../results/tables/seed0 ../results/tables/seed1 ../results/tables/seed2 ../results/tables/seed3 ../results/tables/seed4 \
    --budget 10 --out ../results/tables/cross_seed/host_validity_sensitivity.csv
```

---

## Methodology Notes

**Validity constraint.** Every candidate perturbation is checked against a syntactic-validity predicate (length 4–2,048, valid percent-encoding, parseable with `urllib.parse.urlsplit`, non-empty well-formed host, TLD-pattern preservation except for the domain/path/TLD family) *before* being counted as an edit. The scheme is not validated and the predicate is not a strict RFC 3986 grammar validator. Invalid candidates are discarded and resampled without consuming the edit budget or a classifier query. Every ASR figure in this paper reflects syntactic validity only; DNS-resolution validity was not implemented.

**Query accounting.** One edit = one accepted, applied transformation = one classifier query. Integrated Gradients computation is a white-box operation and is not counted as a classifier query (`src/attacks/attribution.py`, §5.1).

**The `none` defense condition** is a separately trained standard-training control checkpoint produced by `train_defended.py`, not the original target checkpoint from `train.py`. This is why its faithfulness change (paper Table 10) is non-zero despite applying no defense mechanism.

**Faithfulness is measured on the clean classifier** with no attack involved, so the faithfulness–efficiency correlation is non-circular by construction.

**Paired test validity.** The shared evaluation sample frozen in Step 4 ensures that every attack method operates on exactly the same set of URLs, which is required for McNemar's and Wilcoxon's paired tests (paper Table 6) to be valid.

---

## Notes on Results

The `results/` folder contains pre-computed outputs for all five seeds. These can be used directly to reproduce the tables and figures in the paper without re-running the full pipeline, which requires trained model checkpoints (~hours of compute per seed on Apple Silicon MPS).

Model checkpoints (`.pt` files) and the surrogate model's own attack outputs (needed to re-run Step 7) are not included due to size. Contact the author if you need them for verification.

---

## Citation

Citation details will be added upon publication.

Repository: https://github.com/mdhasanmehedi/XAI-adversarial

---

## License

Code released for research reproducibility. Please cite the paper if you use this repository. The dataset is distributed by its Kaggle author under CC0.
