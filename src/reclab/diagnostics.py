"""离线评估诊断：本项目最"像算法工程师"的部分。

四类偏差，四个可被复现的数字：
1. 切分偏差 —— 随机切分 / 留一法 / 时间切分，同一模型指标能差多少；
2. 采样偏差 —— 100 个候选里算 Recall 与全库算 Recall 的差距；
3. 流行度偏差 —— 覆盖率、基尼、长尾占比；
4. 冷启动偏差 —— 按用户活跃度分层后，模型排名是否还成立。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def user_activity_bins(train: pd.DataFrame, bins=(0, 5, 20, 50, 10**9)) -> pd.Series:
    cnt = train.groupby("user_id").size()
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
    return pd.cut(cnt, bins=bins, labels=labels, right=False)


def stratified_recall(
    recs: np.ndarray,
    users: list[int],
    gt: dict[int, set[int]],
    k: int,
    user_bins: pd.Series,
) -> pd.DataFrame:
    """按用户活跃度分层看 Recall@k —— 平均值会掩盖"新用户没人管"。"""
    rows = {}
    for i, u in enumerate(users):
        b = user_bins.get(u)
        if b is None or (isinstance(b, float) and np.isnan(b)):
            b = "unknown"
        rel = gt[u]
        rows.setdefault(str(b), []).append(
            len(rel & set(recs[i, :k].tolist())) / len(rel) if rel else 0.0
        )
    out = [
        {"bucket": b, "n_users": len(v), "recall": float(np.mean(v))}
        for b, v in rows.items()
    ]
    return pd.DataFrame(out).sort_values("bucket").reset_index(drop=True)


def sampled_candidate_recall(
    scores: np.ndarray,
    users: list[int],
    item_ids: np.ndarray,
    gt: dict[int, set[int]],
    k: int,
    n_neg: int = 99,
    seed: int = 0,
) -> float:
    """从全库分数里只留 1 个真 + n_neg 个假，重算 Recall@k。

    这是工业界最常用的"离线评估"，也是最容易高估的评估方式。
    """
    rng = np.random.default_rng(seed)
    hit, total = 0, 0
    for row, u in enumerate(users):
        rel = gt[u]
        pos = [it for it in rel if it in set(item_ids.tolist())]
        if not pos:
            continue
        positives = set(pos)
        pool = np.array([i for i in range(scores.shape[1]) if item_ids[i] not in positives])
        negs = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
        cand = np.concatenate([np.array([np.where(item_ids == p)[0][0] for p in pos]), negs])
        sub = scores[row, cand]
        top = cand[np.argsort(-sub)[:k]]
        hit += len(set(item_ids[top].tolist()) & positives)
        total += len(positives)
    return hit / total if total else float("nan")


def exposure_summary(recs: np.ndarray, item_popularity: np.ndarray, n_items: int, k: int) -> dict:
    from .metrics import coverage_at_k, gini_exposure, long_tail_share

    flat = recs[:, :k].ravel()
    counts = np.bincount(flat, minlength=n_items).astype(float)
    p = counts / counts.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "coverage": round(coverage_at_k(recs, k, n_items), 4),
        "gini": round(gini_exposure(recs, k), 4),
        "long_tail_share": round(long_tail_share(recs, k, item_popularity), 4),
        "exposure_entropy": round(ent, 4),
        "n_distinct_items": int((counts > 0).sum()),
    }


def histogram_bias(pred: np.ndarray, bins: int = 20) -> np.ndarray:
    h, _ = np.histogram(pred, bins=bins, range=(0, 1), density=True)
    return h