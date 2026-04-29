"""
TASK B: FEATURE ENGINEERING SCRIPT
------------------------------------
What it does: Builds a fully numeric feature matrix from consolidated_data.parquet
for binary severity classification.
Goal: Produce taskB/task_b_features.parquet — one row per report with:
  - Demographics (age, sex)
  - Polypharmacy index (num_drugs_taken)
  - Top-15 active substance flags
  - Interaction risk features derived from Task A association rules
  - TARGET: is_severe_outcome

No scaling or imputation is applied here. Both belong inside each model's
sklearn.Pipeline in the modeling script (3B_modeling.py).
NaN in patientonsetage is preserved so tree-based and linear models can
handle it with their respective strategies.
"""

import pandas as pd
import glob
import ast
import os

PARQUET_PATH = "consolidated_data.parquet"
RULES_GLOB   = "taskA/filtered_*.csv"
OUTPUT_PATH  = "taskB/task_b_features.parquet"
LIFT_THRESHOLD = 2.0
TOP_N_DRUGS    = 15


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_frozenset_str(s):
    """Convert 'frozenset({...})' string to a Python set."""
    s = s.strip()
    if s.startswith("frozenset("):
        s = s[len("frozenset("):-1]
    return ast.literal_eval(s)


def _extract_substance(drug_entry):
    """
    Extract active substance name from a FAERS drug dict.
    FAERS stores activesubstance as either a dict or a list — handle both.
    """
    if not isinstance(drug_entry, dict):
        return None
    active = drug_entry.get('activesubstance')
    if isinstance(active, list) and active:
        return active[0].get('activesubstancename')
    elif isinstance(active, dict):
        return active.get('activesubstancename')
    return None


# ── Feature builders ───────────────────────────────────────────────────────────

def _build_target(df):
    """
    is_severe_outcome = 1 if the report involved death, hospitalization,
    or a life-threatening event; 0 otherwise.
    """
    target_cols = [
        'seriousnessdeath',
        'seriousnesshospitalization',
        'seriousnesslifethreatening',
    ]
    present = [c for c in target_cols if c in df.columns]
    severe = pd.Series(False, index=df.index)
    for col in present:
        severe |= (df[col] == '1')
    return severe.astype('int8')


def _build_demographics(df):
    # json_normalize prefixes nested keys — age lives under 'patient.patientonsetage'
    age = pd.to_numeric(
        df.get('patient.patientonsetage', pd.Series(dtype=str, index=df.index)),
        errors='coerce'
    ).clip(0, 120)

    sex = df.get('patient.patientsex', pd.Series(dtype=str, index=df.index))
    return pd.DataFrame({
        'patientonsetage': age,
        'is_male':         (sex == '1').astype('int8'),
        'is_female':       (sex == '2').astype('int8'),
        'sex_unknown':     (~sex.isin(['1', '2'])).astype('int8'),
    }, index=df.index)


def _build_drug_features(df, top_n):
    """
    Returns (poly_series, flags_df, drug_sets_series).
    drug_sets_series is indexed by safetyreportid — used for interaction matching.
    """
    drugs_exp = df[['safetyreportid', 'patient.drug']].explode('patient.drug').copy()
    drugs_exp['substance'] = (
        drugs_exp['patient.drug']
        .apply(_extract_substance)
        .str.upper()
        .str.strip()
    )
    drugs_exp = drugs_exp.dropna(subset=['substance'])

    # Polypharmacy index
    poly = (
        drugs_exp.groupby('safetyreportid')['substance']
        .nunique()
        .rename('num_drugs_taken')
    )

    # Top-N substance boolean flags
    top_drugs = drugs_exp['substance'].value_counts().head(top_n).index.tolist()
    print(f"  Top-{top_n} substances: {top_drugs}")

    flags = {}
    for drug in top_drugs:
        col = 'takes_' + drug.lower().replace(' ', '_').replace('-', '_')
        reports_with = set(drugs_exp.loc[drugs_exp['substance'] == drug, 'safetyreportid'])
        flags[col] = df['safetyreportid'].isin(reports_with).astype('int8').values

    flags_df = pd.DataFrame(flags, index=df.index)

    # Drug set per report (for interaction matching)
    drug_sets = drugs_exp.groupby('safetyreportid')['substance'].apply(set)

    return poly, flags_df, drug_sets


def _build_interaction_features(report_ids, drug_sets, rules_glob, lift_threshold):
    """
    Returns a DataFrame with interaction risk features aligned to report_ids.
    Splits Task A rules into drug-drug and drug-reaction categories using the
    set of all known active substance names as the discriminator.
    """
    rule_files = glob.glob(rules_glob)
    zeros = pd.DataFrame({
        'has_drug_drug_interaction': pd.array([0] * len(report_ids), dtype='int8'),
        'has_drug_reaction_rule':   pd.array([0] * len(report_ids), dtype='int8'),
        'num_matching_rules':       [0] * len(report_ids),
        'max_interaction_lift':     [0.0] * len(report_ids),
    }, index=report_ids.index)

    if not rule_files:
        print(f"  Warning: no rule files found at {rules_glob} — interaction features set to 0.")
        return zeros

    rules_df = pd.concat([pd.read_csv(f) for f in rule_files], ignore_index=True)
    strong   = rules_df[rules_df['lift'] >= lift_threshold].copy()

    if strong.empty:
        print(f"  Warning: no rules with lift >= {lift_threshold}.")
        return zeros

    strong['ant_set'] = strong['antecedents'].apply(_parse_frozenset_str)
    strong['con_set'] = strong['consequents'].apply(_parse_frozenset_str)

    # All known active substance names — used to tell drugs apart from MedDRA reactions
    all_known_drugs = set().union(*drug_sets) if len(drug_sets) > 0 else set()

    strong['is_drug_drug'] = strong.apply(
        lambda r: r['ant_set'] <= all_known_drugs and r['con_set'] <= all_known_drugs,
        axis=1
    )
    strong['is_drug_reaction'] = strong.apply(
        lambda r: r['ant_set'] <= all_known_drugs and not (r['con_set'] <= all_known_drugs),
        axis=1
    )

    dd_ants     = list(strong.loc[strong['is_drug_drug'],     'ant_set'])
    dr_ants     = list(strong.loc[strong['is_drug_reaction'], 'ant_set'])
    all_ant_lift = list(zip(strong['ant_set'], strong['lift']))

    print(f"  Strong rules (lift >= {lift_threshold}): {len(strong)}")
    print(f"  Drug-drug: {len(dd_ants)}  |  Drug-reaction: {len(dr_ants)}")
    print("  Computing per-report interaction features...")

    has_dd, has_dr, num_match, max_lift = [], [], [], []
    for rid in report_ids:
        dset = drug_sets.get(rid, set())
        has_dd.append(int(any(ant <= dset for ant in dd_ants)))
        has_dr.append(int(any(ant <= dset for ant in dr_ants)))
        matching_lifts = [lift for ant, lift in all_ant_lift if ant <= dset]
        num_match.append(len(matching_lifts))
        max_lift.append(max(matching_lifts) if matching_lifts else 0.0)

    return pd.DataFrame({
        'has_drug_drug_interaction': pd.array(has_dd,    dtype='int8'),
        'has_drug_reaction_rule':   pd.array(has_dr,    dtype='int8'),
        'num_matching_rules':       num_match,
        'max_interaction_lift':     max_lift,
    }, index=report_ids.index)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_features(
    parquet_path   = PARQUET_PATH,
    rules_glob     = RULES_GLOB,
    output_path    = OUTPUT_PATH,
    lift_threshold = LIFT_THRESHOLD,
    top_n          = TOP_N_DRUGS,
):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} reports loaded.\n")

    # ── A. Target ──────────────────────────────────────────────────────────
    print("A. Building target variable...")
    target = _build_target(df)
    print(f"  Severe outcomes: {target.sum():,} / {len(target):,}\n")

    # ── B. Demographics ────────────────────────────────────────────────────
    print("B. Extracting demographics...")
    demo = _build_demographics(df)
    print(f"  Age NaN: {demo['patientonsetage'].isna().sum():,}\n")

    # ── C+D. Drug features ─────────────────────────────────────────────────
    print("C+D. Building polypharmacy index and top-substance flags...")
    poly, flags_df, drug_sets = _build_drug_features(df, top_n)
    print()

    # ── E. Interaction features ────────────────────────────────────────────
    print("E. Building interaction features from Task A rules...")
    interactions = _build_interaction_features(
        df['safetyreportid'], drug_sets, rules_glob, lift_threshold
    )
    print()

    # ── F. Assemble & save ─────────────────────────────────────────────────
    print("F. Assembling final feature matrix...")
    base = pd.concat([demo, target.rename('is_severe_outcome')], axis=1)
    base['num_drugs_taken'] = poly.reindex(df['safetyreportid'].values).values
    base['num_drugs_taken'] = base['num_drugs_taken'].fillna(0).astype(int)

    result = pd.concat([base, flags_df, interactions], axis=1)
    result.index = df['safetyreportid']
    result.index.name = 'safetyreportid'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path, engine='pyarrow')

    counts = result['is_severe_outcome'].value_counts()
    total  = len(result)
    print(f"Saved: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  is_severe_outcome  0 (Low Risk):  {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"  is_severe_outcome  1 (High Risk): {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"  NaN in patientonsetage: {result['patientonsetage'].isna().sum():,}")
    print(f"\nColumns ({result.shape[1]}):\n  {list(result.columns)}")

    return result


if __name__ == "__main__":
    build_features()
