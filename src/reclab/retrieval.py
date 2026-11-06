"""召回层：Popularity / ItemCF / BPR-MF / Two-Tower。

统一接口：
    model.fit(train_df, n_users, n_items, ...)
    model.score(users, ...) -> np.ndarray  (len(users), n_items)
再由 `rank_from_scores` 掩掉用户训练集里已交互的物品，取 top-k。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from .config import EMB_DIM, RANDOM_STATE, TORCH_BATCH, TORCH_EPOCHS


# ------------------------------------------------------------------ 工具 ----
def build_index(df: pd.DataFrame) -> tuple[dict, dict]:
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    return {u: i for i, u in enumerate(users)}, {it: i for i, it in enumerate(items)}


def user_history(df: pd.DataFrame, uidx: dict, iidx: dict) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(len(uidx))]
    for u, it in zip(df["user_id"], df["item_id"]):
        if u in uidx and it in iidx:
            out[uidx[u]].append(iidx[it])
    return out


def rank_from_scores(
    scores: np.ndarray,
    users: list[int],
    uidx: dict,
    history: list[list[int]],
    k: int,
) -> np.ndarray:
    """把已交互物品置 -inf 后取 top-k，返回 (n_users, k) 的 item 下标。"""
    s = scores.copy()
    for row, u in enumerate(users):
        ui = uidx.get(u)
        if ui is not None and history[ui]:
            s[row, history[ui]] = -np.inf
    k = min(k, s.shape[1])
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(s, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)


# ------------------------------------------------------------ 1. 热门度 ----
class PopularityRecommender:
    name = "popularity"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, **_) -> "PopularityRecommender":
        self.pop = np.zeros(n_items, dtype=np.float32)
        cnt = train["item_id"].value_counts()
        self.item_ids = np.array(sorted(train["item_id"].unique()))
        self.pop[: len(self.item_ids)] = cnt.reindex(self.item_ids).values.astype(np.float32)
        return self

    def score(self, users, uidx, n_items, **_) -> np.ndarray:
        return np.tile(self.pop, (len(users), 1))


# --------------------------------------------------------------- 2. ItemCF ----
class ItemCFRecommender:
    name = "itemcf"

    def __init__(self, n_neighbors: int = 200):
        self.n_neighbors = n_neighbors

    def fit(self, train, n_users, n_items, uidx=None, iidx=None, **_):
        self.uidx, self.iidx = uidx, iidx
        rows = train["user_id"].map(uidx).values
        cols = train["item_id"].map(iidx).values
        R = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
        )
        # 列归一化后做余弦相似度
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        R = R.multiply(sparse.csr_matrix(1.0 / norms)).tocsr()
        sim = (R.T @ R).toarray()          # (n_items, n_items)
        np.fill_diagonal(sim, 0.0)
        # 只保留每个物品的 top-N 邻居，压成稀疏矩阵
        keep = np.argpartition(-sim, self.n_neighbors, axis=1)[:, : self.n_neighbors]
        mask = np.zeros_like(sim, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        sim = np.where(mask, sim, 0.0).astype(np.float32)
        self.sim = sparse.csr_matrix(sim)
        self.R = R.tocsr()
        return self

    def score(self, users, uidx, n_items, history=None, **_) -> np.ndarray:
        rows = [uidx[u] for u in users]
        sub = self.R[rows]                       # (n_users, n_items)
        return (sub @ self.sim).toarray()


# -------------------------------------------------------------- 3. BPR-MF ----
class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = EMB_DIM):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items, dim)
        nn.init.normal_(self.u.weight, std=0.05)
        nn.init.normal_(self.i.weight, std=0.05)

    def forward(self, u, i):
        return (self.u(u) * self.i(i)).sum(-1)

    @torch.no_grad()
    def all_scores(self, users: torch.Tensor) -> torch.Tensor:
        return self.u(users) @ self.i.weight.T


def train_bpr(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    uidx: dict,
    iidx: dict,
    dim: int = EMB_DIM,
    epochs: int = TORCH_EPOCHS,
    batch: int = TORCH_BATCH,
    lr: float = 3e-3,
    seed: int = RANDOM_STATE,
) -> BPRMF:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BPRMF(n_users, n_items, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    u_arr = train["user_id"].map(uidx).values.astype(np.int64)
    i_arr = train["item_id"].map(iidx).values.astype(np.int64)
    n = len(u_arr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            sel = perm[start : start + batch]
            u = torch.from_numpy(u_arr[sel])
            i = torch.from_numpy(i_arr[sel])
            j = torch.from_numpy(rng.integers(0, n_items, size=len(sel)).astype(np.int64))
            pos = model(u, i)
            neg = model(u, j)
            loss = -F.logsigmoid(pos - neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
    return model


# ------------------------------------------------------------ 4. Two-Tower ----
class TwoTower(nn.Module):
    """用户塔 = ID 向量 + 历史物品向量均值（可处理新用户，给冷启动留了口子）。

    物品塔 = ID 向量 + 流派 one-hot 投影。
    训练用 in-batch softmax（batch 内其它物品当负例），比逐条负采样更稳。
    """

    def __init__(self, n_users: int, n_items: int, n_genres: int, dim: int = EMB_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.genre_proj = nn.Linear(n_genres, dim, bias=False)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        # 温度固定为 0.07：可学习温度在 in-batch softmax 里会一路降到极小值
        # （softmax 变成 one-hot、梯度消失），实测召回反而更差，因此写死。
        self.register_buffer("log_temp", torch.tensor(np.log(0.07)))

    def item_vec(self, item_ids: torch.Tensor, genre: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_emb(item_ids) + self.genre_proj(genre), dim=-1)

    @torch.no_grad()
    def all_item_matrix(self, genre_tensor: torch.Tensor) -> torch.Tensor:
        """全库物品向量矩阵，供历史聚合与打分共用（必须和 item_vec 同空间）。"""
        return self.item_emb.weight + self.genre_proj(genre_tensor)

    def user_vec(self, user_ids: torch.Tensor, hist_mean: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_emb(user_ids) + hist_mean, dim=-1)

    @torch.no_grad()
    def all_scores(self, users, hist_means, item_genres):
        u = self.user_vec(users, hist_means)
        v = self.item_vec(torch.arange(item_genres.shape[0]), item_genres)
        return (u @ v.T) / torch.exp(self.log_temp)


def train_two_tower(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    uidx: dict,
    iidx: dict,
    genre_tensor: torch.Tensor,
    dim: int = EMB_DIM,
    epochs: int = TORCH_EPOCHS,
    batch: int = TORCH_BATCH,
    lr: float = 3e-3,
    seed: int = RANDOM_STATE,
) -> TwoTower:
    torch.manual_seed(seed)
    model = TwoTower(n_users, n_items, genre_tensor.shape[1], dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 行归一化的历史聚合矩阵 H：H[u] @ item_emb = 该用户历史物品向量均值
    hist = user_history(train, uidx, iidx)
    rows, cols = [], []
    for u, h in enumerate(hist):
        rows.extend([u] * len(h))
        cols.extend(h)
    data = np.ones(len(rows), dtype=np.float32)
    H = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    deg = np.asarray(H.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    H = sparse.diags(1.0 / deg) @ H
    H = H.tocsr()

    def epoch_hist_means() -> torch.Tensor:
        """每个 epoch 开始算一次全体用户的历史均值（稀疏矩阵一次乘完）。

        关键：聚合的必须是 item_emb + genre_proj(genre) 的完整物品向量，
        否则用户塔和物品塔不在同一个空间，召回的相似度没有意义。
        """
        with torch.no_grad():
            full = model.all_item_matrix(genre_tensor).numpy()
            return torch.from_numpy(np.asarray(H @ full, dtype=np.float32))

    u_arr = train["user_id"].map(uidx).values.astype(np.int64)
    i_arr = train["item_id"].map(iidx).values.astype(np.int64)
    n = len(u_arr)
    # logQ 校正：in-batch 负例是按曝光（互动次数）采样的，热门物品被当成负例的
    # 概率天然更高，直接做 softmax 会把热门物品的向量压小。减去 log q(j) 相当于
    # 把"采样分布"除回去，是双塔召回的标准做法。
    cnt = np.bincount(i_arr, minlength=n_items).astype(np.float64)
    logq = torch.tensor(
        np.log(np.maximum(cnt, 1.0) / max(cnt.sum(), 1.0)), dtype=torch.float32
    )

    rng = np.random.default_rng(seed)
    t_start = __import__("time").time()
    for ep in range(epochs):
        hist_means = epoch_hist_means()
        perm = rng.permutation(n)
        ep_loss = 0.0
        for start in range(0, n, batch):
            sel = perm[start : start + batch]
            u = torch.from_numpy(u_arr[sel])
            i = torch.from_numpy(i_arr[sel])
            uv = model.user_vec(u, hist_means[u])
            iv = model.item_vec(i, genre_tensor[i])
            logits = (uv @ iv.T) / torch.exp(model.log_temp) - logq[i].unsqueeze(0)
            loss = F.cross_entropy(logits, torch.arange(len(sel)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(sel)
        print(
            f"    two-tower epoch {ep + 1}/{epochs} loss={ep_loss / n:.4f} "
            f"temp={float(torch.exp(model.log_temp)):.3f} "
            f"elapsed={__import__('time').time() - t_start:.1f}s",
            flush=True,
        )
    return model


# ------------------------------------------------------------- 统一评测 ----
def scores_to_item_ids(
    ranked: np.ndarray, users: list[int], uidx: dict, iidx: dict
) -> np.ndarray:
    inv = {v: k for k, v in iidx.items()}
    return np.vectorize(lambda x: inv[x])(ranked)


def evaluate_retrieval(
    score_fn,
    users: list[int],
    gt: dict[int, set[int]],
    uidx: dict,
    iidx: dict,
    k: int,
) -> tuple[np.ndarray, list[int]]:
    """返回 (推荐物品 id 矩阵, 用户列表)，用户按 gt 的 key 排序。"""
    users_eval = [u for u in sorted(gt) if u in uidx]
    scores = score_fn(users_eval)
    hist = None
    return scores, users_eval