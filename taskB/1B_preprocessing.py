"""
=============================================================================
TASK B — HANDOFF NOTES FOR THE MODELING STEP
=============================================================================

WHAT HAS BEEN DONE
------------------
The preprocessing pipeline (steps 1B and 2B) is complete and produces a
ready-to-use feature matrix at:

    taskB/task_b_features.parquet   (108,000 rows x 25 columns)

Run order:
    1. python taskB/1B_dataIngestion.py "taskB/9 json files - no tocar.zip"
       → produces: consolidated_data.parquet  (108,000 rows, 40 raw columns)

    2. python taskB/2B_preprocessing.py
       → produces: taskB/task_b_features.parquet  (108,000 rows, 25 features)

Both scripts are idempotent — safe to re-run, they overwrite outputs.


FEATURE MATRIX DESCRIPTION
---------------------------
Index : safetyreportid (unique FDA report ID)

  DEMOGRAPHICS
    patientonsetage     float  Patient age in years. ~40% NaN — see NaN note below.
    is_male             int8   1 if patient sex reported as male, else 0.
    is_female           int8   1 if patient sex reported as female, else 0.
    sex_unknown         int8   1 if sex not reported, else 0.

  POLYPHARMACY
    num_drugs_taken     int    Count of unique active substances in the report.
                               Higher values correlate with interaction risk.

  TOP-15 ACTIVE SUBSTANCE FLAGS  (int8, 0/1)
    takes_tirzepatide, takes_dupilumab, takes_prednisone, takes_methotrexate,
    takes_omalizumab, takes_adalimumab, takes_acetaminophen, takes_infliximab_dyyb,
    takes_infliximab, takes_rituximab, takes_leuprolide_acetate, takes_tocilizumab,
    takes_vedolizumab, takes_amlodipine_besylate, takes_cetirizine_hydrochloride
    → Computed from the 15 most frequent active substances in the full dataset.

  INTERACTION RISK  (derived from Task A association rules, lift >= 2.0)
    has_drug_drug_interaction  int8   1 if the report contains a drug combination
                                      that matches a high-lift drug-drug rule from
                                      Task A (e.g. DOXORUBICIN + CYCLOPHOSPHAMIDE).
    has_drug_reaction_rule     int8   1 if the report's drug set matches a rule whose
                                      consequent is a MedDRA adverse reaction term.
                                      Most direct pharmacovigilance signal.
    num_matching_rules         int    How many Task A rules are triggered by this report.
    max_interaction_lift       float  Highest lift value among all triggered rules.
                                      0.0 if no rules match.

  TARGET
    is_severe_outcome   int8   BINARY LABEL.
                               1 = report involved death, hospitalization, or
                                   life-threatening event  →  26.6%  (28,683 reports)
                               0 = other outcome           →  73.4%  (79,317 reports)


CLASS BALANCE
-------------
    Severe   (1):  28,683  (26.6%)
    Non-severe (0): 79,317  (73.4%)
Moderate imbalance — manageable without aggressive resampling, but worth
addressing. Recommended: use class_weight='balanced' in sklearn estimators
that support it, or evaluate with F1/AUC-ROC rather than raw accuracy.


NaN HANDLING — IMPORTANT
-------------------------
patientonsetage has ~43,355 NaN values (40% of reports). FAERS does not
require age to be reported. This column was intentionally left as NaN
so each model can handle it appropriately inside its own Pipeline:

    Tree-based models (Random Forest, XGBoost):
        → Handle NaN natively or use SimpleImputer(strategy='median')

    Linear models / SVM / KNN:
        → Require imputation BEFORE scaling:
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('model',   LogisticRegression()),
            ])

All other columns are fully numeric with no missing values.


SCALING
-------
NO scaling has been applied in this script. Raw numeric values are stored.
Apply StandardScaler (or MinMaxScaler) inside sklearn.Pipeline per model:
    - Required for: Logistic Regression, SVM, KNN
    - Not needed for: Random Forest, Decision Tree, XGBoost/LightGBM


NEXT STEP — 3B_modeling.py
---------------------------
Load the feature matrix and train classifiers to predict is_severe_outcome.

Suggested structure:

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, roc_auc_score

    df = pd.read_parquet('taskB/task_b_features.parquet')
    X = df.drop(columns=['is_severe_outcome'])
    y = df['is_severe_outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Random Forest (no scaling needed, handles NaN via imputation) ---
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   RandomForestClassifier(n_estimators=100,
                                           class_weight='balanced',
                                           random_state=42)),
    ])

    # --- Logistic Regression (needs imputation + scaling) ---
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(class_weight='balanced',
                                       max_iter=1000)),
    ])

    # Evaluate with AUC-ROC and F1 (accuracy misleads on imbalanced data)
    for name, pipe in [('Random Forest', rf_pipeline), ('Logistic Reg', lr_pipeline)]:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(f"\\n=== {name} ===")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]):.4f}")
"""
