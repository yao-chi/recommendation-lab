"""评测指标：召回 / 排序 / 排序质量 / 概率校准 / 曝光公平性。

全部是纯函数，配单元测试（tests/test_metrics.py）。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 召回 ----
def recall_at_k(recs: np.ndarray, gt: dict[int, set[int]], k: int) -> float:
    """recs: (n_users, n_items) 推荐列表；gt: user -> 相关物品集合。"""
    hits = 0
    total = 0
    for i, u in enumerate(sorted(gt)):
        rel = gt[u]
        total += len(rel)
        hits += len(rel & set(recs[i, :k].tolist()))
    return hits / total if total else float("nan")


def hit_rate_at_k(recs: np.ndarray, gt: dict[int, set[int]], k: int) -> float:
    users = sorted(gt)
    hit = sum(1 for i, u in enumerate(users) if set(recs[i, :k].tolist()) & gt[u])
    return hit / len(users)


def ndcg_at_k(recs: np.ndarray, gt: dict[int, set[int]], k: int) -> float:
    users = sorted(gt)
    vals = []
    for i, u in enumerate(users):
        rel = gt[u]
        dcg = sum(
            1.0 / np.log2(pos + 2)
            for pos, item in enumerate(recs[i, :k].tolist())
            if item in rel
        )
        ideal = sum(1.0 / np.log2(j + 2) for j in range(min(k, len(rel))))
        vals.append(dcg / ideal if ideal else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def mrr_at_k(recs: np.ndarray, gt: dict[int, set[int]], k: int) -> float:
    users = sorted(gt)
    vals = []
    for i, u in enumerate(users):
        rel = gt[u]
        rr = 0.0
        for pos, item in enumerate(recs[i, :k].tolist()):
            if item in rel:
                rr = 1.0 / (pos + 1)
                break
        vals.append(rr)
    return float(np.mean(vals)) if vals else float("nan")


def coverage_at_k(recs: np.ndarray, k: int, n_items: int) -> float:
    """推荐列表里去重后的物品数 / 全库物品数：越高说明越不"只推头部"。"""
    return len(set(recs[:, :k].ravel().tolist())) / n_items


def gini_exposure(recs: np.ndarray, k: int) -> float:
    """推荐曝光分布的基尼系数：0 完全平均，1 完全集中。"""
    counts = np.bincount(recs[:, :k].ravel(), minlength=int(recs.max()) + 1).astype(float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return float("nan")
    x = np.sort(counts)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


def long_tail_share(recs: np.ndarray, k: int, item_popularity: np.ndarray) -> float:
    """推荐里"长尾物品"(全局交互数处于后 80%)的占比。"""
    thresh = np.quantile(item_popularity[item_popularity > 0], 0.8)
    flat = recs[:, :k].ravel()
    tail = item_popularity[flat] <= thresh
    return float(tail.mean())


# ------------------------------------------------------------ CTR 指标 ----
def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U（含并列 0.5 处理）
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # 处理并列：取平均秩
    all_scores = np.concatenate([pos, neg])
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg_rank = sum_ranks / counts
    ranks = avg_rank[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """期望校准误差：|预测概率 - 实际频率| 按样本量加权的均值。"""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(total)


def reliability_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y_true[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """对用户级指标的均值做 bootstrap 区间。"""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def per_user_ndcg(recs: np.ndarray, gt: dict[int, set[int]], k: int) -> np.ndarray:
    users = sorted(gt)
    out = []
    for i, u in enumerate(users):
        rel = gt[u]
        dcg = sum(
            1.0 / np.log2(pos + 2)
            for pos, item in enumerate(recs[i, :k].tolist())
            if item in rel
        )
        ideal = sum(1.0 / np.log2(j + 2) for j in range(min(k, len(rel))))
        out.append(dcg / ideal if ideal else 0.0)
    return np.array(out)