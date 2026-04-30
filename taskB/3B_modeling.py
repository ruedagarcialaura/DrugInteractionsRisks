"""
TASK B: CLASSIFICATION MODELING SCRIPT — v3
--------------------------------------------
Models: Logistic Regression, Random Forest, XGBoost (GPU), CatBoost (GPU),
        Soft Voting Ensemble (RF + XGB + CatBoost)
New vs v2: reaction features in matrix, GPU-accelerated boosting, ensemble.

Usage:
    python taskB/3B_modeling.py          # full dataset
    python taskB/3B_modeling.py --quick  # 20k sample

Outputs → taskB/TaskBPlots/ + taskB/task_b_evaluation.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, f1_score,
    precision_recall_curve, average_precision_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH = "taskB/task_b_features.parquet"
OUTPUT_DIR    = "taskB/TaskBPlots"
EVAL_CSV      = "taskB/task_b_evaluation.csv"
RESULTS_LOG   = "taskB/results_history.csv"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_FRAC      = 0.10


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(features_path, quick_mode=False):
    print(f"Loading {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"  Shape: {df.shape}")

    X = df.drop(columns=["is_severe_outcome"])
    y = df["is_severe_outcome"]

    counts = y.value_counts()
    total  = len(y)
    print(f"  Severe (1): {counts.get(1,0):,} ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  Non-sev(0): {counts.get(0,0):,} ({counts.get(0,0)/total*100:.1f}%)\n")

    if quick_mode:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=20_000, random_state=RANDOM_STATE)
        idx, _ = next(sss.split(X, y))
        X, y = X.iloc[idx].copy(), y.iloc[idx].copy()
        print(f"  [Quick] Sampled: {X.shape}\n")

    return X, y


def find_best_threshold(y_val, y_prob_val):
    thresholds = np.linspace(0.10, 0.70, 61)
    best_t, best_f1 = 0.50, 0.0
    for t in thresholds:
        pred = (y_prob_val >= t).astype(int)
        f1 = f1_score(y_val, pred, pos_label=1, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def build_pipelines(neg_count, pos_count):
    """
    Returns dict of named sklearn pipelines.
    neg_count / pos_count used to set class weights for gradient boosting models.
    """
    spw = neg_count / pos_count   # scale_pos_weight for XGBoost

    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            class_weight="balanced", max_iter=2000,
            solver="lbfgs", C=0.5, random_state=RANDOM_STATE,
        )),
    ])

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            max_features="sqrt", min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])

    # ── XGBoost ────────────────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        xgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   XGBClassifier(
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                scale_pos_weight=spw,
                tree_method="hist",
                device="cuda",
                eval_metric="auc",
                verbosity=0,
                random_state=RANDOM_STATE,
            )),
        ])
        print("  XGBoost loaded (GPU: cuda)")
    except ImportError:
        xgb_pipe = None
        print("  XGBoost not installed — skipping")

    # ── CatBoost ───────────────────────────────────────────────────────────────
    try:
        from catboost import CatBoostClassifier
        cb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                auto_class_weights="Balanced",
                task_type="GPU",
                devices="0",
                random_seed=RANDOM_STATE,
                verbose=0,
            )),
        ])
        print("  CatBoost loaded (GPU)")
    except ImportError:
        cb_pipe = None
        print("  CatBoost not installed — skipping")

    pipelines = {"Logistic Regression": lr_pipe, "Random Forest": rf_pipe}
    if xgb_pipe is not None:
        pipelines["XGBoost (GPU)"] = xgb_pipe
    if cb_pipe is not None:
        pipelines["CatBoost (GPU)"] = cb_pipe

    return pipelines


def train_evaluate(name, pipe, X_tr, X_v, X_te, y_tr, y_v, y_te):
    print(f"\n{'='*55}\n  Training: {name}\n{'='*55}")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"  Fit time: {elapsed:.1f}s")

    y_prob_val = pipe.predict_proba(X_v)[:, 1]
    best_t = find_best_threshold(y_v, y_prob_val)
    print(f"  Optimal threshold: {best_t:.2f}")

    y_prob      = pipe.predict_proba(X_te)[:, 1]
    y_pred      = (y_prob >= best_t).astype(int)
    auc_roc     = roc_auc_score(y_te, y_prob)
    ap_score    = average_precision_score(y_te, y_prob)
    f1_w        = f1_score(y_te, y_pred, average='weighted')
    f1_sev      = f1_score(y_te, y_pred, pos_label=1, average='binary')
    recall_sev  = float(y_pred[y_te == 1].mean()) if (y_te == 1).sum() else 0.0
    prec_sev    = float(y_te[y_pred == 1].mean()) if (y_pred == 1).sum() else 0.0
    cm          = confusion_matrix(y_te, y_pred, normalize='true')
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_te, y_prob)

    print(f"  AUC-ROC: {auc_roc:.4f}  AP: {ap_score:.4f}  F1-sev: {f1_sev:.4f}  Recall-sev: {recall_sev:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Non-Severe", "Severe"]))

    return dict(
        name=name, pipe=pipe, fit_time=elapsed, threshold=best_t,
        y_pred=y_pred, y_prob=y_prob,
        auc_roc=auc_roc, ap_score=ap_score,
        f1_w=f1_w, f1_sev=f1_sev,
        recall_sev=recall_sev, prec_sev=prec_sev,
        cm=cm, fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),
        prec_c=prec_c, rec_c=rec_c,
    )


def build_ensemble(results, X_v, X_te, y_v, y_te):
    """
    Soft voting ensemble: average predict_proba of all passed models.
    Uses already-trained pipelines — no re-training needed.
    """
    print(f"\n{'='*55}\n  Building Soft Voting Ensemble\n{'='*55}")
    members = [r['name'] for r in results]
    print(f"  Members: {members}")

    # Average probabilities on val set for threshold tuning
    val_probs  = np.mean([r['pipe'].predict_proba(X_v)[:, 1]  for r in results], axis=0)
    test_probs = np.mean([r['pipe'].predict_proba(X_te)[:, 1] for r in results], axis=0)

    best_t = find_best_threshold(y_v, val_probs)
    y_pred = (test_probs >= best_t).astype(int)

    auc_roc    = roc_auc_score(y_te, test_probs)
    ap_score   = average_precision_score(y_te, test_probs)
    f1_w       = f1_score(y_te, y_pred, average='weighted')
    f1_sev     = f1_score(y_te, y_pred, pos_label=1, average='binary')
    recall_sev = float(y_pred[y_te == 1].mean()) if (y_te == 1).sum() else 0.0
    prec_sev   = float(y_te[y_pred == 1].mean())  if (y_pred == 1).sum() else 0.0
    cm         = confusion_matrix(y_te, y_pred, normalize='true')
    fpr, tpr, _ = roc_curve(y_te, test_probs)
    prec_c, rec_c, _ = precision_recall_curve(y_te, test_probs)

    print(f"  Threshold: {best_t:.2f}")
    print(f"  AUC-ROC: {auc_roc:.4f}  AP: {ap_score:.4f}  F1-sev: {f1_sev:.4f}  Recall-sev: {recall_sev:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Non-Severe", "Severe"]))

    return dict(
        name="Voting Ensemble", pipe=None, fit_time=0.0, threshold=best_t,
        y_pred=y_pred, y_prob=test_probs,
        auc_roc=auc_roc, ap_score=ap_score,
        f1_w=f1_w, f1_sev=f1_sev,
        recall_sev=recall_sev, prec_sev=prec_sev,
        cm=cm, fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),
        prec_c=prec_c, rec_c=rec_c,
    )


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(results, output_dir):
    sns.set_theme(style="white")
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"],
                                      display_labels=["Non-Severe", "Severe"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(
            f"{res['name']}\nAUC:{res['roc_auc']:.3f} F1-sv:{res['f1_sev']:.3f} Rec-sv:{res['recall_sev']:.3f}",
            fontsize=9, fontweight="bold"
        )

    fig.suptitle("Task B — Normalised Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_roc_curves(results, output_dir):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", len(results))
    plt.figure(figsize=(9, 6))

    for res, color in zip(results, palette):
        plt.plot(res["fpr"], res["tpr"],
                 label=f"{res['name']}  (AUC={res['roc_auc']:.3f})",
                 color=color, linewidth=2)

    plt.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Task B — ROC Curves: Severity Prediction\nFAERS ~108k reports", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def plot_pr_curves(results, output_dir, baseline):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", len(results))
    plt.figure(figsize=(9, 6))

    for res, color in zip(results, palette):
        plt.plot(res["rec_c"], res["prec_c"],
                 label=f"{res['name']}  (AP={res['ap_score']:.3f})",
                 color=color, linewidth=2)

    plt.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
                label=f"Random baseline (AP={baseline:.3f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Task B — Precision-Recall Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def plot_feature_importance(result, feature_names, output_dir, label, top_n=25):
    model = result["pipe"].named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
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
    ax.set_title(f"{result['name']} — Top {top_n} Features\nFAERS Severity Prediction",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"feature_importance_{label}.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def plot_lr_coefficients(lr_result, feature_names, output_dir, top_n=20):
    model  = lr_result["pipe"].named_steps["model"]
    coefs  = model.coef_[0]
    idx    = np.argsort(np.abs(coefs))[::-1][:top_n]
    feats  = [feature_names[i] for i in idx]
    vals   = coefs[idx]

    colors = ["#e74c3c" if c > 0 else "#3498db" for c in vals[::-1]]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(top_n), vals[::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats[::-1], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (red = raises severe risk)", fontsize=11)
    ax.set_title("Logistic Regression — Feature Coefficients", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "lr_coefficients.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


def save_results(results, eval_csv, results_log):
    rows = [{
        "Model":             res["name"],
        "AUC-ROC":           round(res["auc_roc"],    4),
        "Avg Precision":     round(res["ap_score"],   4),
        "Threshold":         round(res["threshold"],  2),
        "F1 (weighted)":     round(res["f1_w"],       4),
        "F1 (severe=1)":     round(res["f1_sev"],     4),
        "Recall (severe)":   round(res["recall_sev"], 4),
        "Prec. (severe)":    round(res["prec_sev"],   4),
        "Fit time (s)":      round(res["fit_time"],   1),
    } for res in results]

    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    df.to_csv(eval_csv, index=False)

    # Append to history log
    import datetime
    df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "a" if os.path.exists(results_log) else "w"
    df.to_csv(results_log, mode=mode, header=(mode == "w"), index=False)

    print(f"\n  Saved: {eval_csv}")
    print("\n" + df.drop(columns=["timestamp", "Fit time (s)"]).to_markdown(index=False))
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main(quick_mode=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_data(FEATURES_PATH, quick_mode=quick_mode)
    feature_names = list(X.columns)
    print(f"Feature count: {len(feature_names)}")

    # ── Splits ────────────────────────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_size = VAL_FRAC / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size,
        random_state=RANDOM_STATE, stratify=y_trainval
    )
    print(f"Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    baseline_pos_rate = pos_count / (neg_count + pos_count)

    # ── Train all models ──────────────────────────────────────────────────────
    print("\nBuilding pipelines...")
    pipelines = build_pipelines(neg_count, pos_count)
    results = []

    for name, pipe in pipelines.items():
        res = train_evaluate(name, pipe, X_train, X_val, X_test, y_train, y_val, y_test)
        results.append(res)

    # ── Voting Ensemble (all models) ──────────────────────────────────────────
    if len(results) >= 2:
        ensemble_members = [r for r in results if r['name'] != "Logistic Regression"]
        if len(ensemble_members) >= 2:
            ens = build_ensemble(ensemble_members, X_val, X_test, y_val, y_test)
            results.append(ens)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n\nGenerating plots...")
    plot_confusion_matrices(results, OUTPUT_DIR)
    plot_roc_curves(results, OUTPUT_DIR)
    plot_pr_curves(results, OUTPUT_DIR, baseline_pos_rate)

    for res in results:
        if res["pipe"] is not None and "Forest" in res["name"]:
            plot_feature_importance(res, feature_names, OUTPUT_DIR, "rf")
        elif res["pipe"] is not None and "XGBoost" in res["name"]:
            plot_feature_importance(res, feature_names, OUTPUT_DIR, "xgb")
        elif res["pipe"] is not None and "CatBoost" in res["name"]:
            plot_feature_importance(res, feature_names, OUTPUT_DIR, "catboost")
        elif res["pipe"] is not None and "Logistic" in res["name"]:
            plot_lr_coefficients(res, feature_names, OUTPUT_DIR)

    save_results(results, EVAL_CSV, RESULTS_LOG)

    print("\n" + "="*55)
    print("  Task B v3 Complete!")
    best = max(results, key=lambda r: r["auc_roc"])
    print(f"  Best: {best['name']}  AUC={best['auc_roc']:.4f}  F1-sev={best['f1_sev']:.4f}")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick_mode=args.quick)
