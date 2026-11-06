"""三种切分方式：留一法 / 全局时间切分 / 随机切分。

它们回答的是同一个模型的同一个指标能差多少 —— 这是本项目"评估诊断"的第一块。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


def _eligible(train: pd.DataFrame, test: pd.DataFrame) -> list[int]:
    """测试用户必须同时出现在训练集和测试集里，否则无法召回。"""
    return sorted(set(train["user_id"]) & set(test["user_id"]))


def leave_one_out(df: pd.DataFrame, min_interactions: int = 20):
    """每个用户最后一条交互做测试，倒数第二条做验证（序列推荐的工业标准口径）。"""
    d = df.sort_values(["user_id", "ts", "item_id"])
    cnt = d.groupby("user_id")["item_id"].transform("size")
    d = d[cnt >= min_interactions]
    rank_from_end = d.groupby("user_id").cumcount(ascending=False)
    test = d[rank_from_end == 0]
    val = d[rank_from_end == 1]
    train = d[rank_from_end >= 2]
    return train, val, test


def time_split(df: pd.DataFrame, test_frac: float = 0.1):
    """按全局时间戳切分：训练只用过去，测试在未来的时间窗内。"""
    d = df.sort_values("ts")
    cut = int(len(d) * (1 - test_frac))
    cut_ts = d["ts"].iloc[cut]
    train = d[d["ts"] < cut_ts]
    test = d[d["ts"] >= cut_ts]
    users = set(train["user_id"])
    test = test[test["user_id"].isin(users)]
    return train, test, cut_ts


def random_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = RANDOM_STATE):
    """纯随机切分 —— 只用来量化"未来信息泄漏"把指标抬高了多少。"""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_frac
    train = df[~mask]
    test = df[mask]
    users = set(train["user_id"])
    test = test[test["user_id"].isin(users)]
    return train, test


def ground_truth(test: pd.DataFrame) -> dict[int, set[int]]:
    """user_id -> 测试集物品集合（去重）。"""
    gt: dict[int, set[int]] = {}
    for u, g in test.groupby("user_id")["item_id"]:
        gt[int(u)] = set(int(x) for x in g)
    return gt


def next_item_targets(window: pd.DataFrame) -> pd.DataFrame:
    """时间切分窗内，每个用户最早的一条交互 = "下一个会看什么"。

    全局时间切分后如果直接拿整个窗口当 ground truth，重度用户会有上百个目标，
    Recall@20 被稀释到没法比较。取窗口首条交互，等价于"切分点之后的第一步"，
    与留一法（最后一步）口径一致，才能公平对比。
    """
    return window.sort_values("ts").drop_duplicates("user_id", keep="first")


def one_target_per_user(test: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """每个用户随机留一个测试物品 —— 只用于三种切分之间的同口径对比。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test))
    return test.iloc[order].drop_duplicates("user_id", keep="first")


def split_summary(train: pd.DataFrame, test: pd.DataFrame, name: str) -> dict:
    return {
        "split": name,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_users": int(train["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "train_items": int(train["item_id"].nunique()),
        "test_items_unseen_in_train": int(
            len(set(test["item_id"]) - set(train["item_id"]))
        ),
    }