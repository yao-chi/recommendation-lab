"""全部图表（英文轴标签，避免中文字体依赖）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)

C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig, name: str) -> Path:
    p = FIG_DIR / name
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_data_overview(ratings: pd.DataFrame, per_item: pd.Series, path="01_data_overview.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist([ratings["rating"].value_counts().sort_index().values], bins=5)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    counts = ratings["rating"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color=C[0])
    axes[0].set_title("Rating distribution")
    axes[0].set_xlabel("rating")
    axes[0].set_ylabel("count")

    axes[1].hist(np.log10(per_item.values), bins=40, color=C[1])
    axes[1].set_title("Item popularity (log10)")
    axes[1].set_xlabel("log10(interactions)")

    per_user = ratings.groupby("user_id").size()
    axes[2].hist(np.log10(per_user.values), bins=40, color=C[2])
    axes[2].set_title("User activity (log10)")
    axes[2].set_xlabel("log10(interactions)")
    return _save(fig, path)


def fig_recall_curves(res: pd.DataFrame, path="02_recall_curves.png"):
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for i, (name, g) in enumerate(res.groupby("model")):
        g = g.sort_values("k")
        ax.plot(g["k"], g["recall"], marker="o", label=name, color=C[i % len(C)])
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Retrieval: Recall@K (full catalog)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_split_bias(res: pd.DataFrame, path="03_split_bias.png"):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    order = ["random", "leave-one-out", "time"]
    res = res.set_index("split").reindex([s for s in order if s in set(res["split"])]).reset_index()
    bars = ax.bar(res["split"], res["recall20"], color=[C[0], C[1], C[2]])
    for b, v in zip(bars, res["recall20"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Recall@20")
    ax.set_title("Same model, same data, different splits")
    return _save(fig, path)


def fig_sampling_bias(res: pd.DataFrame, path="04_sampling_bias.png"):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = np.arange(len(res))
    w = 0.36
    ax.bar(x - w / 2, res["full_catalog"], width=w, label="full catalog", color=C[0])
    ax.bar(x + w / 2, res["sampled_100"], width=w, label="1 pos + 99 sampled", color=C[3])
    ax.set_xticks(x)
    ax.set_xticklabels(res["model"], rotation=0, fontsize=8)
    ax.set_ylabel("Recall@20")
    ax.set_title("Sampled evaluation inflates recall")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_popularity_bias(expo: pd.DataFrame, path="05_popularity_bias.png"):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(expo))
    w = 0.26
    for i, col in enumerate(["coverage", "gini", "long_tail_share"]):
        ax.bar(x + (i - 1) * w, expo[col], width=w, label=col, color=C[i])
    ax.set_xticks(x)
    ax.set_xticklabels(expo["model"], fontsize=8)
    ax.set_title("Exposure fairness of top-20 lists")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_funnel(funnel: dict, path="06_funnel.png"):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    names = ["Recall@100\n(retrieval)", "NDCG@10\n(ranking)", "NDCG@10\n(reranked)"]
    vals = [funnel.get(k, 0) for k in ("recall100", "ndcg10_rank", "ndcg10_final")]
    bars = ax.bar(names, vals, color=[C[0], C[1], C[2]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    ax.set_title("Two-stage funnel")
    return _save(fig, path)


def fig_roc(roc_data: dict, path="07_ctr_roc.png"):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=C[i % len(C)])
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("CTR models: ROC")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def fig_calibration(cal_data: dict, path="08_calibration.png"):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for i, (name, (xs, ys, _)) in enumerate(cal_data.items()):
        ax.plot(xs, ys, marker="o", ms=3, label=name, color=C[i % len(C)])
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability curve)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_cold_start(res: pd.DataFrame, path="09_cold_start.png"):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    models = res["model"].unique()
    buckets = sorted(res["bucket"].unique())
    x = np.arange(len(buckets))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = res[res["model"] == m].set_index("bucket").reindex(buckets)
        ax.bar(x + (i - len(models) / 2 + 0.5) * w, sub["recall"], width=w, label=m, color=C[i % len(C)])
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=8)
    ax.set_xlabel("user activity bucket (train interactions)")
    ax.set_ylabel("Recall@20")
    ax.set_title("Cold-start stratification")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_ctr_metrics(res: pd.DataFrame, path="10_ctr_metrics.png"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(res))
    axes[0].bar(x, res["auc"], color=C[0])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(res["model"], fontsize=8)
    axes[0].set_ylim(min(res["auc"]) - 0.02, max(res["auc"]) + 0.01)
    axes[0].set_title("AUC")
    for i, v in enumerate(res["auc"]):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    axes[1].bar(x, res["ece"], color=C[3])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(res["model"], fontsize=8)
    axes[1].set_title("ECE (lower is better)")
    for i, v in enumerate(res["ece"]):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    return _save(fig, path)