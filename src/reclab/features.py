"""排序层特征工程：把 (user, item) 对变成 CTR 模型可吃的特征。

设计原则（面试可讲）：
1. 用户侧统计量只用训练集算，避免未来信息泄漏；
2. 物品侧用全库静态属性 + 训练集流行度；
3. 交叉特征抓「这个人对这个类型有多偏好」，这是 GBDT/FM 才吃得下的信号；
4. 稠密特征给线性/FM 的数值通路，稀疏 ID 特征给嵌入通路。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_NEG_EVAL, RANDOM_STATE


# ------------------------------------------------------------ 统计特征 ----
def user_stats(train: pd.DataFrame, users_meta: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    stats = pd.DataFrame(
        {
            "u_cnt": g.size(),
            "u_mean_rating": g["rating"].mean(),
            "u_std_rating": g["rating"].std().fillna(0.0),
        }
    )
    ts = g["ts"].agg(["min", "max"])
    stats["u_span_days"] = (ts["max"] - ts["min"]).dt.total_seconds() / 86400.0
    stats["u_rate_per_day"] = stats["u_cnt"] / stats["u_span_days"].clip(lower=1.0)
    stats = stats.join(users_meta.set_index("user_id"), how="left")
    return stats


def item_stats(train: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    stats = pd.DataFrame(
        {
            "i_cnt": g.size(),
            "i_mean_rating": g["rating"].mean(),
            "i_std_rating": g["rating"].std().fillna(0.0),
        }
    )
    first_seen = g["ts"].min()
    stats["i_first_seen_days"] = (
        (first_seen - train["ts"].min()).dt.total_seconds() / 86400.0
    )
    stats = stats.join(meta.set_index("item_id")[["year", "n_genres"]], how="left")
    return stats


# ------------------------------------------------------------ 样本构造 ----
def build_ctr_pairs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    meta: pd.DataFrame,
    n_users: int,
    n_neg: int = N_NEG_EVAL,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """每个测试用户：最多 5 个正例 + n_neg 个负例。

    负例从"训练集没见过、也不是该用户测试正例"的物品里采样。
    """
    rng = np.random.default_rng(seed)
    valid_items = np.array(sorted(set(meta["item_id"]) & set(train["item_id"])))
    valid_set = set(valid_items.tolist())
    seen = train.groupby("user_id")["item_id"].agg(set).to_dict()
    test_by_user = test.groupby("user_id")["item_id"].apply(list).to_dict()

    users = [u for u in sorted(test_by_user) if u in seen and test_by_user[u]]
    rng.shuffle(users)
    users = users[:n_users]

    rows: list[tuple[int, int, int]] = []
    for u in users:
        pos_all = [int(i) for i in test_by_user[u] if int(i) in valid_set]
        pos = pos_all[:5]
        if not pos:
            continue
        hist = seen[u]
        neg_pool = valid_items[~np.isin(valid_items, list(hist))]
        neg_pool = np.array([i for i in neg_pool if i not in set(pos)])
        if len(neg_pool) == 0:
            continue
        negs = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False)
        rows.extend((u, it, 1) for it in pos)
        rows.extend((u, int(it), 0) for it in negs)

    return pd.DataFrame(rows, columns=["user_id", "item_id", "label"])


# ------------------------------------------------------------ 特征拼接 ----
def add_features(
    pairs: pd.DataFrame,
    train: pd.DataFrame,
    meta: pd.DataFrame,
    genres: list[str],
    users_meta: pd.DataFrame,
    retrieval: pd.DataFrame | None = None,
) -> pd.DataFrame:
    us = user_stats(train, users_meta)
    it = item_stats(train, meta)
    df = pairs.join(us, on="user_id").join(it, on="item_id")

    gcols = [f"g_{g}" for g in genres]
    item_g = meta.set_index("item_id")[genres].add_prefix("g_")
    # 物品流派列（后续算交叉特征 + 取主流派）
    df = df.join(item_g, on="item_id")
    df["genre_top"] = np.asarray(genres)[df[gcols].values.argmax(axis=1)]

    # ---- 交叉特征：用户流派偏好向量 × 物品流派向量 ----
    train_g = train.merge(item_g, on="item_id", how="left")
    pref = train_g.groupby("user_id")[gcols].mean()
    pref = pref.sub(pref.mean(axis=1), axis=0)      # 中心化：只看相对偏好

    p = pref.reindex(df["user_id"]).fillna(0.0).values
    q = item_g.reindex(df["item_id"]).fillna(0.0).values
    df["x_genre_affinity"] = (p * q).sum(axis=1)
    df["x_genre_cosine"] = df["x_genre_affinity"] / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-9
    )
    df["x_pop_x_activity"] = np.log1p(df["i_cnt"]) * np.log1p(df["u_cnt"])
    df["x_item_age"] = df["i_first_seen_days"]

    if retrieval is not None and len(retrieval):
        key = pd.MultiIndex.from_frame(df[["user_id", "item_id"]])
        df["retrieval_score"] = (
            retrieval.set_index(["user_id", "item_id"])["retrieval_score"]
            .reindex(key)
            .fillna(0.0)
            .values
        )
    else:
        df["retrieval_score"] = 0.0

    df["u_cnt"] = np.log1p(df["u_cnt"])
    df["i_cnt"] = np.log1p(df["i_cnt"])
    return df


DENSE_COLS = [
    "u_cnt",
    "u_mean_rating",
    "u_std_rating",
    "u_span_days",
    "u_rate_per_day",
    "i_cnt",
    "i_mean_rating",
    "i_std_rating",
    "i_first_seen_days",
    "year",
    "n_genres",
    "x_genre_affinity",
    "x_genre_cosine",
    "x_pop_x_activity",
    "x_item_age",
    "retrieval_score",
]

SPARSE_COLS = [
    "user_id",
    "item_id",
    "gender",
    "age",
    "occupation",
    "year_bucket",
    "genre_top",
]


def prepare_sparse(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year_bucket"] = (pd.to_numeric(d["year"], errors="coerce") // 10).fillna(-1).astype(int)
    d["age"] = d["age"].astype(int)
    d["occupation"] = d["occupation"].astype(int)
    return d