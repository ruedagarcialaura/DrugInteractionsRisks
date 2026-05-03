"""
TASK B: OPTUNA HYPERPARAMETER TUNING — v1
------------------------------------------
Bayesian hyperparameter optimisation (TPE sampler) for XGBoost and CatBoost.

Strategy:
  - n_estimators ceiling + early_stopping (patience=50) → no direct n_estimators search
  - XGBoost: 100 trials on GPU, searches lr/depth/subsample/colsample/mcw/gamma/alpha/lambda
  - CatBoost: 50 trials on GPU, searches lr/depth/l2/bagging_temp/random_strength/min_data
  - Tuned models re-trained on full train split, evaluated on held-out test split
  - Soft voting ensemble of tuned XGB + tuned CatBoost
  - Threshold tuning on val set (grid 0.10–0.70, 61 steps, maximize F1-severe)
  - Outputs: task_b_tuned_evaluation.csv, optuna_best_params.json,
             TaskBPlots/optuna_history_*.png, TaskBPlots/roc_curves_tuned.png,
             TaskBPlots/feature_importance_xgb_tuned.png

Usage:
    python taskB/4B_optuna_tuning.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH   = "taskB/task_b_features.parquet"
OUTPUT_DIR      = "taskB/TaskBPlots"
TUNED_EVAL_CSV  = "taskB/task_b_tuned_evaluation.csv"
BEST_PARAMS_JSON = "taskB/optuna_best_params.json"
BASELINE_CSV    = "taskB/task_b_evaluation.csv"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
VAL_FRAC        = 0.10

# Optuna trial budgets
XGB_N_TRIALS = 100
CB_N_TRIALS  = 50


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_split(features_path):
    print(f"Loading {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"  Shape: {df.shape}")

    X = df.drop(columns=["is_severe_outcome"])
    y = df["is_severe_outcome"]

    counts = y.value_counts()
    total  = len(y)
    print(f"  Severe (1): {counts.get(1,0):,} ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  Non-sev(0): {counts.get(0,0):,} ({counts.get(0,0)/total*100:.1f}%)\n")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_size = VAL_FRAC / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, random_state=RANDOM_STATE, stratify=y_tv
    )
    print(f"Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}\n")

    # Pre-impute with median (fitted only on train) — no scaling needed for trees
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())

    return (X_train_imp, X_val_imp, X_test_imp,
            y_train.values, y_val.values, y_test.values,
            neg_count, pos_count, list(X.columns))


def find_best_threshold(y_val, y_prob_val):
    thresholds = np.linspace(0.10, 0.70, 61)
    best_t, best_f1 = 0.50, 0.0
    for t in thresholds:
        pred = (y_prob_val >= t).astype(int)
        f1 = f1_score(y_val, pred, pos_label=1, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate_on_test(name, y_prob_val, y_val, y_prob_test, y_test, fit_time):
    best_t = find_best_threshold(y_val, y_prob_val)
    y_pred = (y_prob_test >= best_t).astype(int)

    auc_roc    = roc_auc_score(y_test, y_prob_test)
    ap_score   = average_precision_score(y_test, y_prob_test)
    f1_w       = f1_score(y_test, y_pred, average="weighted")
    f1_sev     = f1_score(y_test, y_pred, pos_label=1, average="binary")
    recall_sev = float(y_pred[y_test == 1].mean()) if (y_test == 1).sum() else 0.0
    prec_sev   = float(y_test[y_pred == 1].mean())  if (y_pred == 1).sum() else 0.0
    cm         = confusion_matrix(y_test, y_pred, normalize="true")
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob_test)

    print(f"\n[{name}]")
    print(f"  AUC-ROC: {auc_roc:.4f}  AP: {ap_score:.4f}  F1-sev: {f1_sev:.4f}  Recall-sev: {recall_sev:.4f}  Threshold: {best_t:.2f}")
    print(classification_report(y_test, y_pred, target_names=["Non-Severe", "Severe"]))

    return dict(
        name=name, fit_time=fit_time, threshold=best_t,
        y_pred=y_pred, y_prob=y_prob_test,
        auc_roc=auc_roc, ap_score=ap_score,
        f1_w=f1_w, f1_sev=f1_sev,
        recall_sev=recall_sev, prec_sev=prec_sev,
        cm=cm, fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),
        prec_c=prec_c, rec_c=rec_c,
    )


# ── XGBoost Optuna study ───────────────────────────────────────────────────────

def run_xgb_study(X_train, X_val, y_train, y_val, neg_count, pos_count, n_trials):
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spw = neg_count / pos_count
    print(f"\n{'='*60}")
    print(f"  XGBoost Optuna Study — {n_trials} trials (GPU)")
    print(f"  scale_pos_weight = {spw:.2f}")
    print(f"{'='*60}")

    def objective(trial):
        params = {
            "n_estimators":     3000,
            "learning_rate":    trial.suggest_float("learning_rate",    0.005, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth",          3, 10),
            "subsample":        trial.suggest_float("subsample",        0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight",   1, 20),
            "gamma":            trial.suggest_float("gamma",            0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
            "scale_pos_weight": spw,
            "tree_method":      "hist",
            "device":           "cuda",
            "eval_metric":      "auc",
            "early_stopping_rounds": 50,
            "verbosity":        0,
            "random_state":     RANDOM_STATE,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best AUC-val: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study


def train_best_xgb(study, X_train, X_val, X_test, y_train, y_val, y_test, neg_count, pos_count):
    from xgboost import XGBClassifier
    spw = neg_count / pos_count
    best = study.best_params.copy()

    print(f"\n  Re-training best XGBoost on full train split...")
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=best["learning_rate"],
        max_depth=best["max_depth"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        min_child_weight=best["min_child_weight"],
        gamma=best["gamma"],
        reg_alpha=best["reg_alpha"],
        reg_lambda=best["reg_lambda"],
        scale_pos_weight=spw,
        tree_method="hist",
        device="cuda",
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    print(f"  Fit time: {elapsed:.1f}s  (best iteration: {model.best_iteration})")

    y_prob_val  = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    res = evaluate_on_test("XGBoost Tuned (GPU)", y_prob_val, y_val, y_prob_test, y_test, elapsed)
    res["model"] = model
    return res


# ── CatBoost Optuna study ──────────────────────────────────────────────────────

def run_cb_study(X_train, X_val, y_train, y_val, n_trials):
    import optuna
    from catboost import CatBoostClassifier, Pool

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n{'='*60}")
    print(f"  CatBoost Optuna Study — {n_trials} trials (GPU)")
    print(f"{'='*60}")

    train_pool = Pool(X_train, y_train)
    val_pool   = Pool(X_val,   y_val)

    def objective(trial):
        params = {
            "iterations":         2000,
            "learning_rate":      trial.suggest_float("learning_rate",    0.01, 0.3, log=True),
            "depth":              trial.suggest_int("depth",              4, 10),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg",      1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength":    trial.suggest_float("random_strength",  0.0, 10.0),
            "min_data_in_leaf":   trial.suggest_int("min_data_in_leaf",   1, 50),
            "auto_class_weights": "Balanced",
            "task_type":          "GPU",
            "devices":            "0",
            "early_stopping_rounds": 50,
            "random_seed":        RANDOM_STATE,
            "verbose":            0,
        }
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best AUC-val: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study


def train_best_cb(study, X_train, X_val, X_test, y_train, y_val, y_test):
    from catboost import CatBoostClassifier, Pool
    best = study.best_params.copy()

    print(f"\n  Re-training best CatBoost on full train split...")
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=best["learning_rate"],
        depth=best["depth"],
        l2_leaf_reg=best["l2_leaf_reg"],
        bagging_temperature=best["bagging_temperature"],
        random_strength=best["random_strength"],
        min_data_in_leaf=best["min_data_in_leaf"],
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        early_stopping_rounds=50,
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    train_pool = Pool(X_train, y_train)
    val_pool   = Pool(X_val,   y_val)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    elapsed = time.time() - t0
    print(f"  Fit time: {elapsed:.1f}s")

    y_prob_val  = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    res = evaluate_on_test("CatBoost Tuned (GPU)", y_prob_val, y_val, y_prob_test, y_test, elapsed)
    res["model"] = model
    return res


# ── Tuned Ensemble ─────────────────────────────────────────────────────────────

def build_tuned_ensemble(xgb_res, cb_res, y_val_prob_xgb, y_val_prob_cb,
                          y_val, y_test):
    print(f"\n{'='*60}\n  Building Tuned Voting Ensemble\n{'='*60}")
    val_probs  = (y_val_prob_xgb + y_val_prob_cb) / 2
    test_probs = (xgb_res["y_prob"] + cb_res["y_prob"]) / 2

    res = evaluate_on_test("Tuned Ensemble", val_probs, y_val, test_probs, y_test, 0.0)
    res["model"] = None
    return res


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_optuna_history(study, title, path):
    """Plot optimization history: trial AUC vs trial number + best-so-far line."""
    trials  = study.trials
    values  = [t.value for t in trials if t.value is not None]
    bests   = [max(values[:i+1]) for i in range(len(values))]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(values)), values, alpha=0.4, s=20, label="Trial AUC", color="#4878CF")
    ax.plot(range(len(bests)),  bests,  linewidth=2, color="#E24A33", label="Best so far")
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Validation AUC-ROC", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves_tuned(results, output_dir, baseline_results=None):
    """ROC curves for tuned models (+optionally baseline for comparison)."""
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", len(results))
    plt.figure(figsize=(9, 6))

    for res, color in zip(results, palette):
        plt.plot(res["fpr"], res["tpr"],
                 label=f"{res['name']}  (AUC={res['roc_auc']:.4f})",
                 color=color, linewidth=2)

    if baseline_results:
        grey = sns.color_palette("Greys", len(baseline_results) + 2)[2:]
        for bres, gc in zip(baseline_results, grey):
            plt.plot(bres["fpr"], bres["tpr"],
                     label=f"{bres['name']} baseline (AUC={bres['roc_auc']:.4f})",
                     color=gc, linewidth=1.5, linestyle="--")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Task B — Tuned Models ROC Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves_tuned.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance_xgb_tuned(model, feature_names, output_dir, top_n=25):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_feat = [feature_names[i] for i in indices]
    top_imp  = importances[indices]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = sns.color_palette("Blues_r", top_n)
    ax.barh(range(top_n), top_imp[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feat[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"XGBoost Tuned — Top {top_n} Features\nFAERS Severity Prediction",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance_xgb_tuned.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(results, best_params, eval_csv, params_json):
    rows = [{
        "Model":           res["name"],
        "AUC-ROC":         round(res["auc_roc"],    4),
        "Avg Precision":   round(res["ap_score"],   4),
        "Threshold":       round(res["threshold"],  2),
        "F1 (weighted)":   round(res["f1_w"],       4),
        "F1 (severe=1)":   round(res["f1_sev"],     4),
        "Recall (severe)": round(res["recall_sev"], 4),
        "Prec. (severe)":  round(res["prec_sev"],   4),
        "Fit time (s)":    round(res["fit_time"],   1),
    } for res in results]

    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    df.to_csv(eval_csv, index=False)
    print(f"\n  Saved: {eval_csv}")

    with open(params_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: {params_json}")

    print("\n" + df.drop(columns=["Fit time (s)"]).to_markdown(index=False))
    return df


def print_comparison(baseline_csv, tuned_df):
    """Print side-by-side delta vs baseline for XGB and CatBoost."""
    if not os.path.exists(baseline_csv):
        return
    base_df = pd.read_csv(baseline_csv)
    print(f"\n{'='*70}")
    print("  BASELINE vs TUNED COMPARISON (AUC-ROC)")
    print(f"{'='*70}")
    for name_map in [("XGBoost (GPU)", "XGBoost Tuned (GPU)"),
                     ("CatBoost (GPU)", "CatBoost Tuned (GPU)"),
                     ("Voting Ensemble", "Tuned Ensemble")]:
        base_name, tuned_name = name_map
        base_row  = base_df[base_df["Model"] == base_name]
        tuned_row = tuned_df[tuned_df["Model"] == tuned_name]
        if base_row.empty or tuned_row.empty:
            continue
        b_auc = base_row["AUC-ROC"].values[0]
        t_auc = tuned_row["AUC-ROC"].values[0]
        delta = t_auc - b_auc
        sign  = "+" if delta >= 0 else ""
        print(f"  {base_name:<22} {b_auc:.4f}  →  {tuned_name:<28} {t_auc:.4f}  (Δ {sign}{delta:.4f})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import optuna
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     neg_count, pos_count,
     feature_names) = load_and_split(FEATURES_PATH)

    # ── 2. XGBoost study ───────────────────────────────────────────────────────
    xgb_study = run_xgb_study(X_train, X_val, y_train, y_val,
                               neg_count, pos_count, XGB_N_TRIALS)
    plot_optuna_history(
        xgb_study,
        f"XGBoost Optuna — {XGB_N_TRIALS} Trials (TPE Sampler, GPU)",
        os.path.join(OUTPUT_DIR, "optuna_history_xgb.png"),
    )

    xgb_res = train_best_xgb(
        xgb_study, X_train, X_val, X_test, y_train, y_val, y_test, neg_count, pos_count
    )

    # ── 3. CatBoost study ──────────────────────────────────────────────────────
    cb_study = run_cb_study(X_train, X_val, y_train, y_val, CB_N_TRIALS)
    plot_optuna_history(
        cb_study,
        f"CatBoost Optuna — {CB_N_TRIALS} Trials (TPE Sampler, GPU)",
        os.path.join(OUTPUT_DIR, "optuna_history_catboost.png"),
    )

    cb_res = train_best_cb(
        cb_study, X_train, X_val, X_test, y_train, y_val, y_test
    )

    # ── 4. Tuned ensemble ──────────────────────────────────────────────────────
    xgb_val_probs = xgb_res["model"].predict_proba(X_val)[:, 1]
    cb_val_probs  = cb_res["model"].predict_proba(X_val)[:, 1]
    ens_res = build_tuned_ensemble(xgb_res, cb_res, xgb_val_probs, cb_val_probs,
                                   y_val, y_test)

    # ── 5. Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    all_results = [xgb_res, cb_res, ens_res]
    plot_roc_curves_tuned(all_results, OUTPUT_DIR)
    plot_feature_importance_xgb_tuned(xgb_res["model"], feature_names, OUTPUT_DIR)

    # ── 6. Save params & CSV ───────────────────────────────────────────────────
    best_params = {
        "xgboost":  xgb_study.best_params,
        "catboost": cb_study.best_params,
    }

    tuned_df = save_results(all_results, best_params, TUNED_EVAL_CSV, BEST_PARAMS_JSON)
    print_comparison(BASELINE_CSV, tuned_df)

    # ── 7. Summary ─────────────────────────────────────────────────────────────
    best = max(all_results, key=lambda r: r["auc_roc"])
    print(f"\n{'='*60}")
    print("  Task B Optuna Tuning Complete!")
    print(f"  Best tuned model: {best['name']}")
    print(f"  AUC-ROC: {best['auc_roc']:.4f}  F1-sev: {best['f1_sev']:.4f}")
    print(f"  XGB best val AUC: {xgb_study.best_value:.4f}")
    print(f"  CB  best val AUC: {cb_study.best_value:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
