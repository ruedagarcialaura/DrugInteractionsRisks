# Predicting High-Risk Drug Interactions in FAERS

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Source: openFDA](https://img.shields.io/badge/Data-openFDA-orange.svg)](https://open.fda.gov/data/faers/)

**CS 584 — Data Mining · Illinois Institute of Technology · Spring 2026**  
Sergio Illescas Cabiró · Carmen Vázquez Pérez de la Cruz · Laura Rueda García

---

## What This Project Does

A full KDD pipeline applied to **108,000 real FDA adverse event reports** (FAERS). Two connected tasks:

- **Task A — Interaction Mining**: Association rule mining (Apriori vs FP-Growth) to discover which drug combinations cause adverse reactions far more often than chance.
- **Task B — Severity Prediction**: Binary classification to predict whether a new patient report will have a severe outcome (death / hospitalization / life-threatening). Task A's rules feed directly into Task B as features.

---

## Final Results

### Task A
| Configuration | Support | Rules found | Max Lift | Avg Confidence |
|---|---|---|---|---|
| Apriori (Active Substances) | 0.0025 | **13** | **223.45** | 0.637 |
| FP-Growth (Active Substances) | 0.0025 | **13** | **223.45** | 0.637 |

Both algorithms produce identical rules — cross-validates correctness. Strongest signal: `LEUCOVORIN + FLUOROURACIL → DIARRHEA` (lift 223×).

### Task B
| Model | AUC-ROC | F1-severe | Recall-severe |
|---|---|---|---|
| Logistic Regression | 0.7519 | 0.533 | 0.641 |
| Random Forest | 0.7916 | 0.570 | 0.644 |
| XGBoost (GPU) | 0.7949 | 0.574 | 0.685 |
| CatBoost (GPU) | 0.7936 | 0.574 | 0.641 |
| Voting Ensemble | 0.7962 | 0.575 | 0.648 |
| **Tuned Ensemble (Optuna ⚡)** | **0.8054** | **0.586** | **0.651** |

The feature `has_drug_reaction_rule` (from Task A) is the **#1 most predictive feature** in the tuned XGBoost classifier.

---

## Repository Structure

```
DrugInteractionsRisks/
│
├── taskA/                        # Task A scripts (run 1A → 8A in order)
│   ├── 1A_dataIngestion.py       # Raw FAERS JSON → consolidated_data.parquet
│   ├── 2A_preprocessing.py       # Transactions (drug+reaction baskets)
│   ├── 3A_Encoding.py            # One-hot binary matrix
│   ├── 4A_APriori.py             # Apriori rule mining
│   ├── 5A_FPGrowth.py            # FP-Growth rule mining
│   ├── 6A_FilterTrueInteractions.py  # Remove indication bias
│   ├── 7A_AnalysisPlots.py       # NetworkX drug-reaction graphs
│   ├── 8A_GenerateMetricsTable.py    # Final metrics CSV
│   └── TaskAPlots/               # Generated network graphs (PNG)
│
├── taskB/                        # Task B scripts
│   ├── 1B_dataIngestion.py       # FAERS ZIP → consolidated_data.parquet
│   ├── 2B_preprocessing.py       # Feature engineering → 71-feature matrix
│   ├── 3B_modeling.py            # Train 5 classifiers, threshold tuning
│   ├── 4B_optuna_tuning.py       # Bayesian hyperparameter search (GPU)
│   ├── optuna_best_params.json   # Best Optuna params (XGBoost + CatBoost)
│   ├── task_b_evaluation.csv     # Baseline model results
│   ├── task_b_tuned_evaluation.csv  # Tuned model results
│   ├── results_history.csv       # Full iteration history
│   └── TaskBPlots/               # ROC curves, confusion matrices, feature importance (PNG)
│
├── TaskB_notebook.ipynb          # End-to-end Task B notebook (code)
├── TaskB_notebook_EXECUTED.ipynb # Same notebook with all outputs rendered
│
├── finalPresentation/
│   └── drug_interactions.html   # Reveal.js final presentation (14 slides)
│
├── local-files/                  # Project documentation and iteration history
│   ├── INDICE.md
│   ├── 00_VISION_GENERAL.md
│   ├── 01_TASK_A_pipeline.md
│   ├── 02_TASK_B_pipeline.md
│   ├── 04_CONCEPTOS_CLAVE.md
│   └── 07_ITERACIONES_TASK_B.md  # Complete Task B iteration history
│
├── dataExploration.py            # Exploratory data analysis plots
├── fields.pdf                    # FAERS field reference
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Data files** (`.json`, `.parquet`, `.zip`) are git-ignored and must be obtained separately. Place the 9 FAERS quarterly JSON files in `taskB/` as a ZIP.

---

## Running Task A

```bash
# 1. Ingest raw FAERS JSON files → consolidated_data.parquet
python taskA/1A_dataIngestion.py <path_to_json_folder>

# 2. Extract drug/reaction baskets
python taskA/2A_preprocessing.py

# 3. One-hot encode
python taskA/3A_Encoding.py

# 4. Mine rules (both algorithms)
python taskA/4A_APriori.py    --file_path taskA/active_substances_encoded.parquet --min_support 0.0025
python taskA/5A_FPGrowth.py   --file_path taskA/active_substances_encoded.parquet --min_support 0.0025

# 5. Filter indication bias
python taskA/6A_FilterTrueInteractions.py --input taskA/association_rules/association_rules_APRIORI_*.csv

# 6. Generate plots
python taskA/7A_AnalysisPlots.py --file taskA/filtered_association_rules/filtered_*.csv

# 7. Generate metrics table
python taskA/8A_GenerateMetricsTable.py
```

Validated `--min_support` value is **0.0025** (0.25%). Lower values cause MemoryErrors or produce 350k+ noisy rules.

---

## Running Task B

**Option A — Notebook (recommended):** Open and run `TaskB_notebook_EXECUTED.ipynb` to see all results, or `TaskB_notebook.ipynb` to re-run from scratch. The notebook handles all steps automatically, including dependency checks and Optuna tuning.

> Run from the project root folder. The notebook asserts `taskA/` exists.

**Option B — Scripts:**

```bash
# 1. Ingest (or reuse consolidated_data.parquet from Task A)
python taskB/1B_dataIngestion.py "taskB/9 json files - no tocar.zip"

# 2. Feature engineering → task_b_features.parquet (71 features)
python taskB/2B_preprocessing.py

# 3. Train baseline models
python taskB/3B_modeling.py

# 4. Optuna hyperparameter tuning (requires GPU for speed)
python taskB/4B_optuna_tuning.py
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Active substance names over brand names | Consolidates brand fragmentation; same lift, less noise |
| min_support = 0.0025 | Empirically validated: lowest threshold where all algorithms terminate and produce interpretable rules |
| Indication bias filter (Step 6A) | Removes rules like `METHOTREXATE → RHEUMATOID ARTHRITIS` (prescribed use, not adverse effect) |
| 70 / 10 / 20 train/val/test split | Val set used only for threshold tuning — never model selection |
| NaN preserved in `patientonsetage` | Global imputation before split = data leakage; each model imputes within its own `sklearn.Pipeline` |
| SVM replaced by XGBoost + CatBoost | SVM collapsed to all-negative predictions (AUC ≈ 0.63) on this imbalanced dataset; gradient boosting handles class imbalance natively |
| Leakage detection | `rxn_hospitalisation` (MedDRA) directly encoded the target variable; removing it dropped AUC 0.805→0.796, confirming real leakage |
| Task A rules as Task B features | `has_drug_reaction_rule` (lift ≥ 2.0 match) became the #1 most predictive feature in the tuned classifier |

---

## Dependencies

| Package | Role |
|---|---|
| `mlxtend` | `TransactionEncoder`, `apriori`, `fpgrowth`, `association_rules` |
| `xgboost` ≥ 2.0 | GPU-accelerated gradient boosting |
| `catboost` ≥ 1.2 | GPU-accelerated boosting with native categorical support |
| `optuna` ≥ 3.0 | Bayesian hyperparameter optimization (TPE sampler) |
| `scikit-learn` | Pipelines, LR, RF, preprocessing, evaluation metrics |
| `pandas` / `pyarrow` | Data manipulation and Parquet I/O |
| `networkx` / `matplotlib` | Drug-reaction network visualization |
