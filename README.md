# Attribution-Guided Evasion of Malicious URL Detectors

Code and results for:

> **Attribution-Guided Evasion of Malicious URL Detectors: A Multi-Seed Study of Faithfulness, Transfer, and Defense**  
> Md Mehedi Hasan — Independent researcher, Bogura, Bangladesh  
> Contact: hasanmd24@outlook.com  
> https://github.com/mdhasanmehedi/XAI-adversarial

---

## Overview

This study investigates whether Integrated Gradients attribution can serve as an effective attack resource against a character-level CNN-Transformer malicious URL detector, and whether explanation faithfulness is statistically associated with adversarial attack efficiency. All experiments are replicated across **five independently trained model seeds** — a design choice that materially revised several conclusions relative to what a single trained model would have suggested.

**Dataset:** 651,191-URL four-class benchmark (benign, phishing, malware, defacement). After exact-duplicate removal: 641,119 unique URLs, split 70/10/20 into train/validation/test.

**Target model:** Character-level CNN-Transformer, 658,629 parameters. Clean accuracy 97.9% ± 0.2%, macro-F1 96.8% ± 0.3% across five seeds.

**Surrogate model:** Architecturally distinct smaller variant (117,541 parameters) used for transfer experiments.

---

## Key Findings

**H1 — Attribution guidance improves attack success rate (strongly supported).**  
One-shot guidance significantly beats the predefined `random_substitution` reference in all five seeds; adaptive guidance in four of five. Success-rate gaps reach 33.6 percentage points (seed 3: 53.2% vs 19.6%). Notably, `domain_path_tld_transform` — the only unguided method that also targets a specific URL region — still achieves only 14.8% ASR, ruling out "any targeted heuristic works equally well" as an explanation.

**H2 — Efficiency gains are weak and inconsistent (weakly supported).**  
Significant edit-cost advantage over the reference in only 2–3 of 5 seeds per method. The near-zero mean effect size masks high instability: per-seed effect sizes range from −0.63 to +0.68, with standard deviations larger than the means themselves.

**H3 — Faithfulness–efficiency association is confounded by class (limited evidence after correction).**  
Naive within-seed correlations pooled across URL classes produce a spurious association. After class adjustment, evidence for a genuine faithfulness–efficiency relationship is limited and heterogeneous. 13 pre-adjustment significant instances (out of 40 seed × method combinations) collapse to 4 after class correction.

**H4 — Transfer is non-trivial but unpredicted by surrogate strength (empirically demonstrated).**  
All seven methods transfer at 29–48% on average across five independently trained targets. Surrogate attack strength does not predict transfer rate: the method with the highest surrogate ASR (42.8%) ranks only fourth in transfer rate, while the weakest surrogate method (10.4%) transfers best of all.

**H5 — Combined defense is the only consistently robust condition (supported descriptively).**  
Combined adversarial training + character-level augmentation reduces attack success in all five seeds. Neither mechanism alone generalizes consistently across independently trained models. The same defense condition is also associated with the largest measured reduction in explanation faithfulness — a robustness–faithfulness trade-off observed as an association, not a statistically established causal claim.

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
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── transforms.py             # Six character-level transform primitives
│   │   ├── validity.py               # RFC 3986 syntactic-validity predicate
│   │   ├── attribution.py            # Integrated Gradients
│   │   ├── build_eval_sample.py      # Freezes the shared 10,000-URL evaluation sample
│   │   ├── unguided.py               # Six unguided baseline attack families
│   │   ├── guided.py                 # One-shot and adaptive IG-guided attacks
│   │   ├── transfer_evaluate.py      # Surrogate-to-target transfer evaluation (Table 5)
│   │   └── compare_guided_vs_unguided.py  # McNemar + Wilcoxon + FDR (Table 3)
│   │
│   ├── faithfulness/
│   │   ├── __init__.py
│   │   ├── deletion_auc.py           # Deletion AUC + Comprehensiveness/Sufficiency
│   │   └── correlate_with_attacks.py # Spearman + bootstrap CI + class adjustment (Table 4)
│   │
│   └── defense/
│       ├── __init__.py
│       ├── train_defended.py         # FGM / charaug / combined / none conditions
│       └── evaluate_defense.py       # Three-attack generalization battery (Tables 6–7)
│
├── results/
│   └── tables/
│       ├── faithfulness_results.csv        # Seed 3 raw per-URL faithfulness (Figures 3–4)
│       ├── guided_attack_results.csv       # Seed 3 raw per-URL guided attacks
│       ├── unguided_attack_results.csv     # Seed 3 raw per-URL unguided attacks
│       ├── cross_seed/                     # Aggregated cross-seed summaries (Tables 2–5)
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

Each `seedN/` folder contains 15 files:

| File | Contents |
|---|---|
| `faithfulness_results.csv` | Per-URL Deletion AUC, Comprehensiveness, Sufficiency |
| `guided_attack_results.csv` | Per-URL one-shot and adaptive attack outcomes |
| `unguided_attack_results.csv` | Per-URL outcomes for all six unguided methods |
| `seedN_table2_results.csv` | ASR and edit cost across budgets (Table 2 / Table 2-S) |
| `seedN_table3_results.csv` | Guided vs. unguided significance test results (Table 3) |
| `seedN_table3_guided_only.csv` | Guided-method budget sweep summary |
| `seedN_table4_by_class.csv` | Faithfulness–efficiency correlations by URL class |
| `seedN_table4_results.csv` | Faithfulness–efficiency correlations overall (Table 4) |
| `seedN_table5_results.csv` | Surrogate transfer results (Table 5) |
| `seedN_table6_results.csv` | Defense evaluation — ASR and robust accuracy (Table 6) |
| `seedN_table7_results.csv` | Defense evaluation — faithfulness change (Table 7) |
| `table6_attack_battery_charaug_seedN.csv` | Defense battery: seen-type attack (charaug) |
| `table6_attack_battery_combined_seedN.csv` | Defense battery: combined defense |
| `table6_attack_battery_fgm_seedN.csv` | Defense battery: FGM |
| `table6_attack_battery_none_seedN.csv` | Defense battery: undefended baseline |

---

## Dataset

The study uses the publicly available `malicious_phish.csv` URL dataset. The raw file is **not included** here — it is available from its original public source and cited in the paper.

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

Experiments were run on Apple Silicon (MPS backend). CUDA and CPU are also supported — `train.py` auto-detects the available device.

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
python attacks/compare_guided_vs_unguided.py --outdir ../results/tables/seedN
python faithfulness/correlate_with_attacks.py --outdir ../results/tables/seedN
```

### 10. Aggregate across seeds
```bash
python aggregate_cross_seed.py \
    --results_base ../results/tables \
    --outdir ../results/tables/cross_seed \
    --seeds 0 1 2 3 4
```

---

## Methodology Notes

**Validity constraint.** Every candidate perturbation is checked against a syntactic-validity predicate (RFC 3986 parsing, well-formed scheme and host, valid percent-encoding, TLD-pattern preservation) *before* being counted as an edit. Invalid candidates are discarded and resampled without consuming the edit budget or a classifier query. Every ASR figure in this paper reflects syntactic validity only; DNS-resolution validity was not implemented.

**Query accounting.** One edit = one accepted, applied transformation = one classifier query. Integrated Gradients computation is a white-box operation and is not counted as a classifier query (`src/attacks/attribution.py`, §5.1).

**The `none` defense condition** is a separately trained standard-training control checkpoint produced by `train_defended.py`, not the original target checkpoint from `train.py`. This is why its faithfulness change in Table 7 is non-zero despite applying no defense mechanism.

**Faithfulness is measured on the clean classifier** with no attack involved, so the faithfulness–efficiency correlation is non-circular by construction.

**Paired test validity.** The shared evaluation sample frozen in Step 4 ensures that every attack method operates on exactly the same set of URLs, which is required for McNemar's and Wilcoxon's paired tests in Table 3 to be valid.

---

## Notes on Results

The `results/` folder contains pre-computed outputs for all five seeds. These can be used directly to reproduce the tables and figures in the paper without re-running the full pipeline, which requires trained model checkpoints (~hours of compute per seed on Apple Silicon MPS).

Model checkpoints (`.pt` files) are not included due to size. Contact the author if you need them for verification.

---

## Citation

Citation details will be added upon publication.

Repository: https://github.com/mdhasanmehedi/XAI-adversarial

---

## License

Code released for research reproducibility. Please cite the paper if you use this repository.
