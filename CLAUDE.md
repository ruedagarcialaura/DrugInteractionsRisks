# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pharmacovigilance research project that mines FDA Adverse Event Reporting System (FAERS) data (~108k+ records from 9 quarterly JSON files) to discover high-risk drug-drug interactions (DDIs). The core technique is association rule mining comparing Apriori and FP-Growth algorithms.

- **Task A**: Interaction mining via association rules (implemented, complete pipeline)
- **Task B**: Severity prediction via classification (ingestion + feature engineering implemented; modeling pending)

## Setup

```bash
pip install -r requirements.txt
```

Data files (`.json`, `.parquet`, `.csv`, `.zip`) are git-ignored and must be obtained separately.

## Pipeline Execution (Task A — run in order)

```bash
# 1. Ingest raw FAERS JSON files → consolidated_data.parquet
python taskA/1A_dataIngestion.py <path_to_json_folder>

# 2. Extract drug/reaction transactions → parquet transaction files
python taskA/2A_preprocessing.py

# 3. One-hot encode transactions → binary matrices
python taskA/3A_Encoding.py

# 4. Run Apriori (or FP-Growth) on encoded data
python taskA/4A_APriori.py --file_path taskA/active_substances_encoded.parquet --min_support 0.0025
python taskA/5A_FPGrowth.py --file_path taskA/active_substances_encoded.parquet --min_support 0.0025

# 5. Filter out noise (indication bias, non-medical consequents)
python taskA/6A_FilterTrueInteractions.py --input taskA/association_rules/association_rules_APRIORI_*.csv

# 6. Visualize as network graphs → taskA/TaskAPlots/*.png
python taskA/7A_AnalysisPlots.py --file taskA/filtered_association_rules/filtered_*.csv

# 7. Generate evaluation metrics table → taskA/evaluation/final_metrics_table.csv
python taskA/8A_GenerateMetricsTable.py

# Optional: Exploratory data analysis plots → plots/
python dataExploration.py
```

The empirically validated `--min_support` value is **0.0025** (0.25%). Lower values cause combinatorial explosion; higher values miss rare but significant interactions.

## Pipeline Execution (Task B — run in order)

Task B requires Task A's filtered association rules (`taskA/filtered_*.csv`) to be present before running step 2.

```bash
# 1. Ingest raw FAERS JSON zip → consolidated_data.parquet (shared with Task A)
python taskB/1B_dataIngestion.py "taskB/9 json files - no tocar.zip"

# 2. Feature engineering → taskB/task_b_features.parquet
python taskB/2B_preprocessing.py
```

`taskB/1B_preprocessing.py` is an empty skeleton superseded by `2B_preprocessing.py`.

## Architecture & Data Flow

```
FAERS JSON files
    → [1A_dataIngestion]      → consolidated_data.parquet  (69 raw columns)
    → [2A_preprocessing]      → task_a_transactions_*.parquet  (transactional baskets)
    → [3A_Encoding]           → *_encoded.parquet  (binary one-hot matrix)
    → [4A_APriori / 5A_FPGrowth] → association_rules/*.csv  (support, confidence, lift)
    → [6A_FilterTrueInteractions] → filtered_association_rules/*.csv
    → [7A_AnalysisPlots]      → TaskAPlots/*.png  (NetworkX graphs)
    → [8A_GenerateMetricsTable] → evaluation/final_metrics_table.csv

consolidated_data.parquet (shared)
    → [2B_preprocessing]      → taskB/task_b_features.parquet  (~25 numeric columns + target)
```

**Key design decisions:**
- Active substance names (generic) are preferred over brand names — consolidates drug redundancy across manufacturers.
- Only 4 of 69 FAERS fields are used: `safetyreportid`, `patient.drug.medicinalproduct`, `patient.drug.activesubstance.activesubstancename`, `patient.reaction.reactionmeddrapt`.
- Parquet is used throughout for efficient storage of nested list structures.
- The filtering step (step 6) removes indication bias — rules like "Methotrexate → Rheumatoid Arthritis" reflect prescribed use, not adverse effects.
- Network graphs use teal nodes for drugs, coral for reactions, edge weight proportional to lift.
- Task B feature matrix has no scaling — apply `StandardScaler` inside `sklearn.Pipeline` per algorithm. `patientonsetage` NaN is preserved for algorithm-specific imputation.
- Task A filtered rules feed Task B: `has_drug_reaction_rule` (lift ≥ 2.0) is the strongest signal, linking known drug combos directly to MedDRA adverse events.
- FAERS `activesubstance` appears as both a dict and a list in the raw JSON — `2B_preprocessing.py` handles both formats; `2A_preprocessing.py` only handles the dict case.

## Key Dependencies

| Package | Role |
|---------|------|
| `mlxtend` | `TransactionEncoder`, `apriori`, `fpgrowth`, `association_rules` |
| `pandas` / `pyarrow` / `fastparquet` | Data manipulation and parquet I/O |
| `networkx` / `matplotlib` | Network graph visualization |
| `tqdm` | Progress bars in mining steps |
| `tabulate` | Markdown table generation in metrics script |

## No Test Suite

There are no automated tests. Validation is done by inspecting intermediate parquet/CSV outputs at each pipeline stage and verifying record counts match expectations (>100k records after ingestion).
