# -*- coding: utf-8 -*-
"""
day05_tree_rf_xgboost_lightgbm.py
===================================
Day 05 — Benchmark: Decision Tree vs. Random Forest vs. XGBoost vs. LightGBM
Dataset : California Housing (20,640 samples, 8 numerical features)
Author  : ML Engineering Series
Date    : 2026-10-25

This script benchmarks four tree-based architectures on a regression task,
logging training time, inference latency, bias-variance gap (Train/Test RMSE),
R² scores, and feature importances. Outputs a dark-themed 2×2 dashboard PNG.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
import io
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatter
from scipy.stats import gaussian_kde

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")             # suppress verbose LGB/XGB logs

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL AESTHETICS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0D1117",
    "surface":   "#161B22",
    "border":    "#30363D",
    "text":      "#E6EDF3",
    "muted":     "#8B949E",
    "DT":        "#FF6B6B",   # Decision Tree — warm red
    "RF":        "#4ECDC4",   # Random Forest — teal
    "XGB":       "#FFE66D",   # XGBoost       — amber
    "LGBM":      "#A78BFA",   # LightGBM      — violet
    "train":     "#64B5F6",   # train bars    — sky blue
    "test":      "#EF5350",   # test bars     — coral
    "accent":    "#58A6FF",
}

MODEL_COLORS = [PALETTE["DT"], PALETTE["RF"], PALETTE["XGB"], PALETTE["LGBM"]]
MODEL_NAMES  = ["Decision\nTree", "Random\nForest", "XGBoost", "LightGBM"]
SHORT_NAMES  = ["DT", "RF", "XGB", "LGBM"]

INFERENCE_BATCH = 1_000          # latency measured per this many samples

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA PREPARATION — strict 80/20 split, no leakage
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split():
    """Load California Housing and return a strict train/test split."""
    data = fetch_california_housing(as_frame=False)
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    print(f"\n{'='*64}")
    print(f"  California Housing Dataset")
    print(f"{'='*64}")
    print(f"  Total samples      : {X.shape[0]:,}")
    print(f"  Features           : {X.shape[1]}  ({', '.join(feature_names)})")
    print(f"  Train samples      : {X_train.shape[0]:,}")
    print(f"  Test  samples      : {X_test.shape[0]:,}")
    print(f"  Target (MedHouseVal): min={y.min():.2f}  max={y.max():.2f}  "
          f"mean={y.mean():.2f}")
    print(f"{'='*64}\n")

    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_models():
    """Return an ordered list of (name, model) tuples."""
    models = [
        (
            "Decision Tree",
            DecisionTreeRegressor(max_depth=12, random_state=42)
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
            )
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=42,
                verbosity=0, eval_metric="rmse"
            )
        ),
        (
            "LightGBM",
            LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=42,
                verbose=-1
            )
        ),
    ]
    return models


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train each model, measure metrics, return structured results dict."""
    results = {}

    for name, model in models:
        print(f"  >>  Training  {name} ...", end="", flush=True)

        # ── Training time ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # ── Predictions ────────────────────────────────────────────────────
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        # ── RMSE (uses squared=False / manual sqrt for compatibility) ──────
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
        test_r2    = r2_score(y_test, y_pred_test)

        # ── Inference latency — time to predict INFERENCE_BATCH samples ────
        batch = X_test[:INFERENCE_BATCH]
        repeats = 20
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model.predict(batch)
        latency_ms = ((time.perf_counter() - t0) / repeats) * 1_000   # ms

        # ── Feature importances ────────────────────────────────────────────
        raw_fi = model.feature_importances_
        fi_norm = raw_fi / raw_fi.sum()

        # ── Residuals ─────────────────────────────────────────────────────
        residuals = y_test - y_pred_test

        results[name] = {
            "model":       model,
            "train_time":  train_time,
            "latency_ms":  latency_ms,
            "train_rmse":  train_rmse,
            "test_rmse":   test_rmse,
            "test_r2":     test_r2,
            "fi_norm":     fi_norm,
            "residuals":   residuals,
            "y_pred_test": y_pred_test,
        }

        print(f"  done  [{train_time:.3f}s | RMSE_test={test_rmse:.4f} | R²={test_r2:.4f}]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TERMINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table(results):
    """Print a ranked leaderboard sorted by Test R²."""
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["test_r2"], reverse=True
    )

    col_w = [20, 10, 10, 12, 12, 18]
    headers = ["Model", "R2 Score", "Test RMSE", "Train RMSE",
               "Train Time", "Latency (ms/1k)"]

    SEP  = "-" * (sum(col_w) + len(col_w) * 3 + 1)
    SEP2 = "=" * 64
    hfmt = " | ".join(f"{{:<{w}}}" for w in col_w)
    rfmt = " | ".join(f"{{:<{w}}}" for w in col_w)

    print(f"\n{SEP2}")
    print(f"  MODEL LEADERBOARD  (ranked by Test R2)")
    print(f"{SEP2}")
    print(f"  {hfmt.format(*headers)}")
    print(f"  {SEP}")

    medals = ["[1]", "[2]", "[3]", "[4]"]
    for rank, (name, m) in enumerate(sorted_results):
        medal = medals[rank] if rank < 4 else "   "
        row = [
            f"{medal} {name}",
            f"{m['test_r2']:.4f}",
            f"{m['test_rmse']:.4f}",
            f"{m['train_rmse']:.4f}",
            f"{m['train_time']:.3f}s",
            f"{m['latency_ms']:.2f} ms",
        ]
        print(f"  {rfmt.format(*row)}")

    print(f"  {SEP}")
    print(f"\n  Overfitting Gap (Train RMSE - Test RMSE):")
    for name, m in results.items():
        gap = m["train_rmse"] - m["test_rmse"]
        bar = "#" * int(abs(gap) * 40)
        sign = "+" if gap > 0 else ""
        print(f"    {name:<16}: {sign}{gap:+.4f}  {bar}")
    print(f"{SEP2}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALIZATION — 2×2 dark-themed dashboard
# ─────────────────────────────────────────────────────────────────────────────
def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["surface"],
        "axes.edgecolor":     PALETTE["border"],
        "axes.labelcolor":    PALETTE["text"],
        "axes.titlecolor":    PALETTE["text"],
        "axes.grid":          True,
        "grid.color":         PALETTE["border"],
        "grid.linewidth":     0.6,
        "grid.alpha":         0.5,
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "text.color":         PALETTE["text"],
        "legend.facecolor":   PALETTE["surface"],
        "legend.edgecolor":   PALETTE["border"],
        "legend.labelcolor":  PALETTE["text"],
        "font.family":        "DejaVu Sans",
        "font.size":          9,
    })


def panel_bias_variance(ax, results):
    """Panel 1 — Side-by-side Train RMSE vs. Test RMSE bar chart."""
    names      = list(results.keys())
    n          = len(names)
    train_rms  = [results[k]["train_rmse"] for k in names]
    test_rms   = [results[k]["test_rmse"]  for k in names]

    x      = np.arange(n)
    width  = 0.35
    offset = width / 2

    bars_train = ax.bar(x - offset, train_rms, width, label="Train RMSE",
                        color=PALETTE["train"], alpha=0.85, zorder=3,
                        edgecolor=PALETTE["bg"], linewidth=0.8)
    bars_test  = ax.bar(x + offset, test_rms,  width, label="Test RMSE",
                        color=PALETTE["test"],  alpha=0.85, zorder=3,
                        edgecolor=PALETTE["bg"], linewidth=0.8)

    # Annotate values
    for bar in bars_train:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7, color=PALETTE["train"], fontweight="bold")
    for bar in bars_test:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7, color=PALETTE["test"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=8.5)
    ax.set_ylabel("RMSE  (lower is better)", fontsize=9)
    ax.set_title("① Bias–Variance Tradeoff", fontsize=11, fontweight="bold",
                 pad=10, color=PALETTE["accent"])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(train_rms) * 1.25)

    # Overfitting gap arrows
    for i, (tr, te) in enumerate(zip(train_rms, test_rms)):
        if te > tr * 1.05:
            ax.annotate("", xy=(i + offset, te), xytext=(i - offset, tr),
                        arrowprops=dict(arrowstyle="->",
                                        color=PALETTE["muted"], lw=0.8))


def panel_efficiency(ax, results):
    """Panel 2 — Training time (log) vs. Test R² bubble/scatter plot."""
    names     = list(results.keys())
    train_t   = np.array([results[k]["train_time"] for k in names])
    r2_scores = np.array([results[k]["test_r2"]    for k in names])
    latency   = np.array([results[k]["latency_ms"] for k in names])

    # bubble size proportional to latency
    sizes = (latency / latency.max()) * 1800 + 200

    for i, (name, color) in enumerate(zip(names, MODEL_COLORS)):
        ax.scatter(train_t[i], r2_scores[i], s=sizes[i],
                   color=color, alpha=0.85, edgecolors=PALETTE["bg"],
                   linewidths=1.5, zorder=4)
        ax.annotate(
            f" {SHORT_NAMES[i]}\n  R²={r2_scores[i]:.3f}\n  {train_t[i]:.2f}s",
            (train_t[i], r2_scores[i]),
            fontsize=7.5, color=color, fontweight="bold",
            xytext=(6, 0), textcoords="offset points", va="center"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Time — seconds  (log scale)", fontsize=9)
    ax.set_ylabel("Test R²  (higher is better)", fontsize=9)
    ax.set_title("② Efficiency Benchmark  (bubble ∝ inference latency)",
                 fontsize=11, fontweight="bold", pad=10, color=PALETTE["accent"])

    # Custom legend for bubble sizes
    legend_sizes = [latency.min(), latency.mean(), latency.max()]
    legend_labels = [f"{v:.1f} ms" for v in legend_sizes]
    legend_bubbles = [plt.scatter([], [], s=(v / latency.max()) * 1800 + 200,
                                  color=PALETTE["muted"], alpha=0.6,
                                  edgecolors=PALETTE["bg"])
                      for v in legend_sizes]
    ax.legend(legend_bubbles, legend_labels, title="Latency / 1k",
              loc="lower right", fontsize=7, title_fontsize=7.5,
              scatterpoints=1)


def panel_residuals(ax, results, y_test):
    """Panel 3 — Residual KDE/histogram: Decision Tree vs. LightGBM."""
    dt_res   = results["Decision Tree"]["residuals"]
    lgbm_res = results["LightGBM"]["residuals"]

    bins = np.linspace(-4.5, 4.5, 70)
    ax.hist(dt_res,   bins=bins, color=PALETTE["DT"],   alpha=0.35,
            label="Decision Tree", density=True, zorder=2)
    ax.hist(lgbm_res, bins=bins, color=PALETTE["LGBM"], alpha=0.35,
            label="LightGBM",      density=True, zorder=2)

    # KDE overlays
    for res, color, name in [
        (dt_res,   PALETTE["DT"],   "DT KDE"),
        (lgbm_res, PALETTE["LGBM"], "LightGBM KDE"),
    ]:
        kde = gaussian_kde(res, bw_method=0.3)
        x_range = np.linspace(-4.5, 4.5, 400)
        ax.plot(x_range, kde(x_range), color=color, lw=2.0,
                label=name, zorder=3)

    ax.axvline(0, color=PALETTE["accent"], lw=1.2, linestyle="--",
               alpha=0.8, label="Zero residual", zorder=4)

    # Std deviation annotations
    for res, color, xoff in [(dt_res, PALETTE["DT"], -2.8),
                              (lgbm_res, PALETTE["LGBM"], 1.0)]:
        ax.text(xoff, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.3,
                f"σ = {res.std():.3f}", color=color, fontsize=8,
                fontweight="bold", ha="center")

    ax.set_xlabel("Residual  (y_true − ŷ)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("③ Residual Distribution — DT vs. LightGBM",
                 fontsize=11, fontweight="bold", pad=10, color=PALETTE["accent"])
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.set_xlim(-4.5, 4.5)


def panel_feature_importance(ax, results, feature_names):
    """Panel 4 — Top-5 normalized feature importance: RF, XGB, LightGBM."""
    models_fi = {
        "Random Forest": results["Random Forest"]["fi_norm"],
        "XGBoost":       results["XGBoost"]["fi_norm"],
        "LightGBM":      results["LightGBM"]["fi_norm"],
    }
    colors_fi = [PALETTE["RF"], PALETTE["XGB"], PALETTE["LGBM"]]

    # Rank features by mean importance across the 3 models
    mean_fi   = np.mean([v for v in models_fi.values()], axis=0)
    top5_idx  = np.argsort(mean_fi)[::-1][:5]
    top5_feat = [feature_names[i] for i in top5_idx]

    x      = np.arange(len(top5_feat))
    n_bars = len(models_fi)
    width  = 0.22
    offsets = np.linspace(-(n_bars - 1) * width / 2,
                          (n_bars - 1) * width / 2, n_bars)

    for (mname, fi), color, offset in zip(models_fi.items(), colors_fi, offsets):
        vals = [fi[i] for i in top5_idx]
        bars = ax.bar(x + offset, vals, width, label=mname,
                      color=color, alpha=0.85, zorder=3,
                      edgecolor=PALETTE["bg"], linewidth=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=6, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(top5_feat, fontsize=9)
    ax.set_ylabel("Normalized Importance", fontsize=9)
    ax.set_title("④ Feature Importance — RF vs. XGB vs. LightGBM (Top 5)",
                 fontsize=11, fontweight="bold", pad=10, color=PALETTE["accent"])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(max(models_fi[m]) for m in models_fi) * 1.35)


def build_dashboard(results, feature_names, y_test, out_path):
    """Compose all 4 panels into a 2×2 dark-themed dashboard."""
    set_dark_style()

    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97, top=0.91, bottom=0.07
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    panel_bias_variance(ax1, results)
    panel_efficiency(ax2, results)
    panel_residuals(ax3, results, y_test)
    panel_feature_importance(ax4, results, feature_names)

    # ── Super-title ────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        "Day 05  ·  Tree Ensemble Benchmark  ·  California Housing (n=20,640)",
        ha="center", va="top", fontsize=15, fontweight="bold",
        color=PALETTE["text"]
    )
    fig.text(
        0.5, 0.942,
        "Decision Tree  |  Random Forest  |  XGBoost  |  LightGBM",
        ha="center", va="top", fontsize=10, color=PALETTE["muted"]
    )

    # ── Color legend for model identity ───────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=c, markersize=10, label=n)
        for c, n in zip(MODEL_COLORS,
                        ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"])
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=9, framealpha=0.3,
               bbox_to_anchor=(0.5, 0.003),
               facecolor=PALETTE["surface"], edgecolor=PALETTE["border"])

    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  [SAVED]  Dashboard saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ARTIFACT_PATH = "day05_tree_ensemble_artifact.png"

    SEP = "=" * 64
    print("\n" + SEP)
    print("  DAY 05 -- Tree Ensemble Benchmark")
    print("  Decision Tree | Random Forest | XGBoost | LightGBM")
    print(SEP)

    # 1. Data
    X_train, X_test, y_train, y_test, feature_names = load_and_split()

    # 2. Build models
    models = build_models()

    # 3. Train & evaluate
    print("  Training models ...\n")
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # 4. Terminal summary
    print_summary_table(results)

    # 5. Feature importance terminal output
    print("  Feature Importance Rankings (normalized):")
    print(f"  {'Feature':<14}", end="")
    ensemble_keys = ["Random Forest", "XGBoost", "LightGBM"]
    for k in ensemble_keys:
        print(f"  {k:>14}", end="")
    print()
    print("  " + "-" * (14 + 3 * 16))

    for i, fname in enumerate(feature_names):
        print(f"  {fname:<14}", end="")
        for k in ensemble_keys:
            val = results[k]["fi_norm"][i]
            print(f"  {val:>14.4f}", end="")
        print()
    print()

    # 6. Dashboard
    print("  Building visualization dashboard...")
    build_dashboard(results, feature_names, y_test, ARTIFACT_PATH)

    SEP = "=" * 64
    print("\n" + SEP)
    print("  [DONE]  Artifact: day05_tree_ensemble_artifact.png")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
