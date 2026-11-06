"""MovieLens-1M 读取与数据体检。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from .config import DATA_DIR


@dataclass
class Dataset:
    ratings: pd.DataFrame      # user_id, item_id, rating, ts
    movies: pd.DataFrame       # item_id, title, genres
    users: pd.DataFrame        # user_id, gender, age, occupation, zip

    @property
    def n_users(self) -> int:
        return int(self.ratings["user_id"].nunique())

    @property
    def n_items(self) -> int:
        return int(self.ratings["item_id"].nunique())

    @property
    def n_interactions(self) -> int:
        return int(len(self.ratings))


def _read_ml(path: str, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / path,
        sep="::",
        engine="python",
        names=names,
        encoding="latin-1",
    )


def load_dataset() -> Dataset:
    ratings = _read_ml("ratings.dat", ["user_id", "item_id", "rating", "ts"])
    ratings["ts"] = pd.to_datetime(ratings["ts"], unit="s")
    movies = _read_ml("movies.dat", ["item_id", "title", "genres"])
    users = _read_ml("users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    return Dataset(ratings=ratings, movies=movies, users=users)


def genre_matrix(movies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """item_id × 18 个流派的 0/1 矩阵（一部电影可属多个流派）。"""
    all_genres = sorted({g for gs in movies["genres"] for g in gs.split("|")})
    rows = []
    for item, gs in zip(movies["item_id"], movies["genres"]):
        row = {"item_id": item}
        row.update({g: int(g in gs.split("|")) for g in all_genres})
        rows.append(row)
    return pd.DataFrame(rows), all_genres


def file_digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_health(ds: Dataset) -> dict:
    """数据体检：重复、评分分布、活跃度长尾、时间跨度。"""
    r = ds.ratings
    dup = int(r.duplicated(["user_id", "item_id"]).sum())
    per_user = r.groupby("user_id").size()
    per_item = r.groupby("item_id").size()
    return {
        "n_interactions": int(len(r)),
        "n_users": ds.n_users,
        "n_items": ds.n_items,
        "duplicate_user_item_pairs": dup,
        "sparsity": round(1 - len(r) / (ds.n_users * ds.n_items), 6),
        "rating_mean": round(float(r["rating"].mean()), 4),
        "rating_std": round(float(r["rating"].std()), 4),
        "ts_min": str(r["ts"].min().date()),
        "ts_max": str(r["ts"].max().date()),
        "user_activity_p50": int(per_user.median()),
        "user_activity_p90": int(per_user.quantile(0.9)),
        "user_activity_max": int(per_user.max()),
        "item_pop_p50": int(per_item.median()),
        "item_pop_p90": int(per_item.quantile(0.9)),
        "item_pop_max": int(per_item.max()),
        # 头部 1% 物品占据的交互份额 —— 流行度偏斜率
        "top1pct_item_share": round(
            float(per_item.sort_values(ascending=False).head(max(1, ds.n_items // 100)).sum() / len(r)), 4
        ),
    }


def item_meta(ds: Dataset) -> pd.DataFrame:
    g, genres = genre_matrix(ds.movies)
    meta = ds.movies.merge(g, on="item_id", how="left")
    meta["year"] = (
        meta["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float")
    )
    meta["year"] = meta["year"].fillna(meta["year"].median())
    meta["n_genres"] = meta[genres].sum(axis=1)
    return meta, genres