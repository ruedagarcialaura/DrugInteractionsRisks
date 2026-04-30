"""
TASK B: CLASSIFICATION MODELING SCRIPT
---------------------------------------
What it does: Trains and evaluates three classifiers (Logistic Regression,
Random Forest, HistGradientBoostingClassifier) on the FAERS feature matrix
to predict adverse event severity. Generates evaluation plots and a comparison
table.

Goal: Predict is_severe_outcome (1 = severe, 0 = non-severe) from patient
demographics, polypharmacy index, and interaction risk features derived
from Task A association rules.

Usage:
    python taskB/3B_modeling.py                  # full dataset
    python taskB/3B_modeling.py --quick-mode     # 20k sample for speed

Outputs (saved to taskB/TaskBPlots/):
    - confusion_matrices.png
    - roc_curves.png
    - pr_curves.png
    - feature_importance_rf.png
    - feature_importance_hgbc.png
    taskB/task_b_evaluation.csv  (comparison table of all models)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH = "taskB/task_b_features.parquet"
OUTPUT_DIR    = "taskB/TaskBPlots"
EVAL_CSV      = "taskB/task_b_evaluation.csv"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_FRAC      = 0.10   # fraction of training set used to tune thresholds


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(features_path, quick_mode=False):
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"  Shape: {df.shape}")

    X = df.drop(columns=["is_severe_outcome"])
    y = df["is_severe_outcome"]

    counts = y.value_counts()
    total  = len(y)
    print(f"  Severe   (1): {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"  Non-sev. (0): {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%)\n")

    if quick_mode:
        print("  [Quick Mode] Sampling 20,000 records (stratified) for speed...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=20_000, random_state=RANDOM_STATE)
        idx, _ = next(sss.split(X, y))
        X = X.iloc[idx].copy()
        y = y.iloc[idx].copy()
        print(f"  Sampled shape: {X.shape}\n")

    return X, y


def find_best_threshold(y_val, y_prob_val):
    """Find the probability threshold that maximises F1 on severe class."""
    thresholds = np.linspace(0.10, 0.70, 61)
    best_t, best_f1 = 0.50, 0.0
    for t in thresholds:
        pred = (y_prob_val >= t).astype(int)
        f1 = f1_score(y_val, pred, pos_label=1, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def build_pipelines():
    """Returns a dict of named sklearn pipelines."""
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            C=0.5,
        )),
    ])

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # HistGradientBoostingClassifier: handles NaN natively, no imputer needed,
    # faster and better than LinearSVC on tabular imbalanced data.
    hgbc_pipe = Pipeline([
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    return {
        "Logistic Regression":      lr_pipe,
        "Random Forest":            rf_pipe,
        "Gradient Boosting (HGBC)": hgbc_pipe,
    }


def train_evaluate(name, pipe, X_train, X_val, X_test, y_train, y_val, y_test):
    """Trains a pipeline, tunes threshold on val set, evaluates on test set."""
    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"{'='*55}")

    pipe.fit(X_train, y_train)

    # Tune threshold on validation set
    y_prob_val = pipe.predict_proba(X_val)[:, 1]
    best_t = find_best_threshold(y_val, y_prob_val)
    print(f"  Optimal threshold (F1-severe on val): {best_t:.2f}")

    # Evaluate on test set using optimal threshold
    y_prob      = pipe.predict_proba(X_test)[:, 1]
    y_pred      = (y_prob >= best_t).astype(int)
    auc_roc     = roc_auc_score(y_test, y_prob)
    f1_w        = f1_score(y_test, y_pred, average="weighted")
    f1_sev      = f1_score(y_test, y_pred, pos_label=1, average="binary")
    recall_sev  = (y_pred[y_test == 1] == 1).mean() if (y_test == 1).sum() else 0.0
    prec_sev    = (y_test[y_pred == 1] == 1).mean() if (y_pred == 1).sum() else 0.0
    ap_score    = average_precision_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred, normalize="true")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    print(f"\n  AUC-ROC:             {auc_roc:.4f}")
    print(f"  Avg Precision (AP):  {ap_score:.4f}")
    print(f"  Threshold used:      {best_t:.2f}")
    print(f"  F1 (weighted):       {f1_w:.4f}")
    print(f"  F1 (severe=1):       {f1_sev:.4f}")
    print(f"  Recall  (severe=1):  {recall_sev:.4f}")
    print(f"  Precision(severe=1): {prec_sev:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Severe", "Severe"]))

    return {
        "name":       name,
        "pipe":       pipe,
        "threshold":  best_t,
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "auc_roc":    auc_roc,
        "ap_score":   ap_score,
        "f1_w":       f1_w,
        "f1_sev":     f1_sev,
        "recall_sev": recall_sev,
        "prec_sev":   prec_sev,
        "cm":         cm,
        "fpr":        fpr,
        "tpr":        tpr,
        "prec_c":     prec_c,
        "rec_c":      rec_c,
        "roc_auc":    roc_auc_val,
    }


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(results, output_dir):
    """Plots side-by-side normalised confusion matrices for all models."""
    sns.set_theme(style="white")
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=["Non-Severe", "Severe"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(
            f"{res['name']}\nAUC: {res['roc_auc']:.3f}  |  F1(severe): {res['f1_sev']:.3f}"
            f"  |  Recall(severe): {res['recall_sev']:.3f}",
            fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "Task B — Normalised Confusion Matrices: Severity Prediction",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close()


def plot_roc_curves(results, output_dir):
    """Plots overlapping ROC curves for all models."""
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", len(results))

    plt.figure(figsize=(8, 6))
    for res, color in zip(results, palette):
        plt.plot(
            res["fpr"], res["tpr"],
            label=f"{res['name']}  (AUC = {res['roc_auc']:.3f})",
            color=color, linewidth=2,
        )

    plt.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(
        "Task B — ROC Curves: Severity Prediction\n"
        "FAERS Dataset (~108k reports, 25 features)",
        fontsize=13, fontweight="bold",
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def plot_pr_curves(results, output_dir, baseline_pos_rate):
    """Plots Precision-Recall curves for all models."""
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", len(results))

    plt.figure(figsize=(8, 6))
    for res, color in zip(results, palette):
        plt.plot(
            res["rec_c"], res["prec_c"],
            label=f"{res['name']}  (AP = {res['ap_score']:.3f})",
            color=color, linewidth=2,
        )

    plt.axhline(y=baseline_pos_rate, color="k", linestyle="--", linewidth=1,
                label=f"Random baseline (AP = {baseline_pos_rate:.3f})")
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title(
        "Task B — Precision-Recall Curves: Severity Prediction\n"
        "Higher curves → better detection of severe cases",
        fontsize=13, fontweight="bold",
    )
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def plot_feature_importance(result, feature_names, output_dir, label="rf", top_n=20):
    """Plots feature importance from tree-based model."""
    model = result["pipe"].named_steps["model"]
    importances = model.feature_importances_
    # Align feature names — HGBC pipeline has no imputer step so names match directly
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("Blues_r", top_n)

    ax.barh(range(top_n), top_values[::-1], color=palette[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini / split gain)", fontsize=11)
    ax.set_title(
        f"{result['name']} — Top {top_n} Feature Importances\n"
        "Predicting Severe Adverse Event Outcomes (FAERS)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(output_dir, f"feature_importance_{label}.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def plot_lr_coefficients(lr_result, feature_names, output_dir, top_n=20):
    """Plots LR coefficients as a diverging bar chart."""
    model = lr_result["pipe"].named_steps["model"]
    # Imputer doesn't change column count; scaler doesn't either
    coefs = model.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1][:top_n]

    top_features = [feature_names[i] for i in sorted_idx]
    top_coefs    = coefs[sorted_idx]

    colors = ["#e74c3c" if c > 0 else "#3498db" for c in top_coefs[::-1]]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top_coefs[::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-odds coefficient (red = raises severe risk)", fontsize=11)
    ax.set_title(
        "Logistic Regression — Feature Coefficients\n"
        "Predicting Severe Adverse Event Outcomes (FAERS)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "lr_coefficients.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def save_comparison_table(results, eval_csv):
    """Saves a CSV comparison table of all models."""
    rows = []
    for res in results:
        rows.append({
            "Model":             res["name"],
            "AUC-ROC":           round(res["auc_roc"],    4),
            "Avg Precision":     round(res["ap_score"],   4),
            "Threshold":         round(res["threshold"],  2),
            "F1 (weighted)":     round(res["f1_w"],       4),
            "F1 (severe=1)":     round(res["f1_sev"],     4),
            "Recall (severe=1)": round(res["recall_sev"], 4),
            "Prec. (severe=1)":  round(res["prec_sev"],   4),
        })

    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    df.to_csv(eval_csv, index=False)
    print(f"\n  Comparison table saved: {eval_csv}")
    print("\n" + df.to_markdown(index=False))
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main(quick_mode=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    X, y = load_data(FEATURES_PATH, quick_mode=quick_mode)
    feature_names = list(X.columns)

    # 2. Train / val / test split (stratified)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    val_size = VAL_FRAC / (1 - TEST_SIZE)          # fraction of trainval for val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    baseline_pos_rate = float(y_train.mean())

    # 3. Build + train + evaluate each pipeline
    pipelines = build_pipelines()
    results   = []

    for name, pipe in pipelines.items():
        res = train_evaluate(name, pipe, X_train, X_val, X_test, y_train, y_val, y_test)
        results.append(res)

    # 4. Generate plots
    print("\n\nGenerating plots...")
    plot_confusion_matrices(results, OUTPUT_DIR)
    plot_roc_curves(results, OUTPUT_DIR)
    plot_pr_curves(results, OUTPUT_DIR, baseline_pos_rate)

    rf_res = next(r for r in results if "Forest" in r["name"])
    plot_feature_importance(rf_res, feature_names, OUTPUT_DIR, label="rf")

    hgbc_res = next(r for r in results if "Gradient" in r["name"])
    # HGBC feature_importances_ index maps to the features seen during fit —
    # since HGBC handles NaN natively there is no imputer step changing the
    # feature count, so feature_names aligns directly.
    plot_feature_importance(hgbc_res, feature_names, OUTPUT_DIR, label="hgbc")

    lr_res = next(r for r in results if "Logistic" in r["name"])
    plot_lr_coefficients(lr_res, feature_names, OUTPUT_DIR)

    # 5. Save comparison table
    save_comparison_table(results, EVAL_CSV)

    print("\n" + "=" * 55)
    print("  Task B Modeling Complete!")
    print(f"  Plots saved to:  {OUTPUT_DIR}/")
    print(f"  Results CSV:     {EVAL_CSV}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task B: Severity classification")
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use a 20k subsample for fast testing (skips full 108k training)",
    )
    args = parser.parse_args()
    main(quick_mode=args.quick_mode)
