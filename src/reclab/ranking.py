"""排序层：LR / FM / DeepFM / GBDT 四种 CTR 模型，同一套特征、同一个评估。

为什么放四种？
- LR：可解释、强基线，工业界 CTR 的起点（Facebook 2014 之前的标配）；
- FM：二阶交叉，解决"特征组合稀疏"；
- DeepFM：FM + 高阶非线性，端到端；
- GBDT：表格特征的现实王者，用来验证"深度模型真的赢了吗"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy import sparse as sp

from .config import RANDOM_STATE
from .features import DENSE_COLS, SPARSE_COLS


# ------------------------------------------------------------ 编码器 ----
class FieldEncoder:
    """稀疏字段 -> 全局 one-hot 下标（LR 用），同时记录每个字段的基数（FM 用）。"""

    def __init__(self) -> None:
        self.maps: dict[str, dict] = {}
        self.offsets: dict[str, int] = {}
        self.cardinalities: list[int] = []

    def fit(self, df: pd.DataFrame) -> "FieldEncoder":
        offset = 0
        for c in SPARSE_COLS:
            vals = sorted(df[c].unique().tolist())
            self.maps[c] = {v: i for i, v in enumerate(vals)}
            self.offsets[c] = offset
            offset += len(vals)
            self.cardinalities.append(len(vals))
        self.total = offset
        return self

    def transform_idx(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(SPARSE_COLS)), dtype=np.int64)
        for j, c in enumerate(SPARSE_COLS):
            m = self.maps[c]
            out[:, j] = df[c].map(lambda v: m.get(v, 0)).values
        return out

    def transform_onehot(self, df: pd.DataFrame) -> sp.csr_matrix:
        idx = self.transform_idx(df)
        rows = np.repeat(np.arange(len(df)), len(SPARSE_COLS))
        cols = idx.ravel() + np.tile(
            np.array([self.offsets[c] for c in SPARSE_COLS]), len(df)
        )
        # 上面 cols 的顺序与 idx.ravel() 对齐（列优先展开）
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(df), self.total),
        )


class Scaler:
    def fit(self, x: np.ndarray) -> "Scaler":
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.nan_to_num(x, nan=0.0) - self.mu) / self.sd
        return np.clip(z, -8, 8).astype(np.float32)


def dense_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(df[DENSE_COLS].to_numpy(dtype=np.float32), nan=0.0)


# ------------------------------------------------------------ 深度模型 ----
class FM(nn.Module):
    def __init__(self, cardinalities: list[int], n_dense: int, k: int = 16):
        super().__init__()
        self.offsets = np.concatenate([[0], np.cumsum(cardinalities)[:-1]])
        self.emb = nn.Embedding(sum(cardinalities), k)
        self.lin = nn.Embedding(sum(cardinalities), 1)
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.zeros_(self.lin.weight)
        self.dense = nn.Linear(n_dense, 1)

    def forward(self, x_sparse: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        idx = x_sparse + torch.as_tensor(self.offsets, device=x_sparse.device)
        v = self.emb(idx)                                   # B, F, k
        linear = self.lin(idx).sum(dim=(1, 2))
        inter = 0.5 * ((v.sum(1) ** 2) - (v ** 2).sum(1)).sum(1)
        return linear + inter + self.dense(x_dense).squeeze(-1)


class DeepFM(nn.Module):
    def __init__(self, cardinalities: list[int], n_dense: int, k: int = 16, hidden=(128, 64)):
        super().__init__()
        self.fm = FM(cardinalities, n_dense, k)
        in_dim = len(cardinalities) * k + n_dense
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_sparse, x_dense):
        idx = x_sparse + torch.as_tensor(self.fm.offsets, device=x_sparse.device)
        v = self.fm.emb(idx).flatten(1)
        return self.fm(x_sparse, x_dense) + self.mlp(
            torch.cat([v, x_dense], dim=1)
        ).squeeze(-1)


def _train_torch(model, x_sp, x_de, y, xv_sp, xv_de, yv, epochs=8, batch=2048, lr=1e-3):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x_sp)
    xd = torch.from_numpy(x_de)
    yt = torch.from_numpy(y.astype(np.float32))
    xvt = torch.from_numpy(xv_sp)
    xvd = torch.from_numpy(xv_de)
    yvt = torch.from_numpy(yv.astype(np.float32))
    n = len(yt)
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            b = perm[s : s + batch]
            opt.zero_grad()
            out = model(xt[b], xd[b])
            loss = lossf(out, yt[b])
            loss.backward()
            opt.step()
        auc = _auc_from_logits(model, xvt, xvd, yvt)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _auc_from_logits(model, x_sp, x_de, y) -> float:
    model.eval()
    logits = model(x_sp, x_de)
    p = torch.sigmoid(logits).numpy()
    y = y.numpy()
    from .metrics import auc_score

    return auc_score(y, p)


@torch.no_grad()
def predict_torch(model, x_sp, x_de) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x_sp), torch.from_numpy(x_de))).numpy()


# ------------------------------------------------------------ 统一训练 ----
def fit_lr(enc: FieldEncoder, scaler: Scaler, tr, va):
    X = sp.hstack(
        [enc.transform_onehot(tr), sp.csr_matrix(scaler.transform(dense_matrix(tr)))]
    ).tocsr()
    Xv = sp.hstack(
        [enc.transform_onehot(va), sp.csr_matrix(scaler.transform(dense_matrix(va)))]
    ).tocsr()
    clf = LogisticRegression(max_iter=300, C=0.5, solver="liblinear")
    clf.fit(X, tr["label"].values)
    return clf.predict_proba(Xv)[:, 1]


def fit_gbdt(enc: FieldEncoder, scaler: Scaler, tr, va):
    """GBDT 基线：稠密特征 + 稀疏字段的整数编码。

    只有低基数字段（<=255 类）交给 HistGradientBoosting 的原生 categorical 分裂；
    user_id / item_id 这种上千类的字段按数值处理，避免触发 sklearn 的基数上限。
    """
    Xtr = np.hstack(
        [scaler.transform(dense_matrix(tr)), enc.transform_idx(tr).astype(np.float32)]
    )
    Xva = np.hstack(
        [scaler.transform(dense_matrix(va)), enc.transform_idx(va).astype(np.float32)]
    )
    cat_idx = [
        len(DENSE_COLS) + j
        for j, card in enumerate(enc.cardinalities)
        if card <= 255
    ]
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        l2_regularization=1.0,
        categorical_features=cat_idx,
        random_state=RANDOM_STATE,
    )
    clf.fit(Xtr, tr["label"].values)
    return clf.predict_proba(Xva)[:, 1]


def fit_fm(enc, scaler, tr, va, deep: bool = False):
    card = enc.cardinalities
    n_dense = len(DENSE_COLS)
    model = DeepFM(card, n_dense) if deep else FM(card, n_dense)
    x_sp = enc.transform_idx(tr)
    x_de = scaler.transform(dense_matrix(tr))
    y = tr["label"].values
    # 验证集切 15% 做早停
    rng = np.random.default_rng(RANDOM_STATE)
    m = rng.random(len(tr)) < 0.85
    xv_sp, xv_de, yv = x_sp[~m], x_de[~m], y[~m]
    _train_torch(
        model, x_sp[m], x_de[m], y[m], xv_sp, xv_de, yv,
        epochs=6 if deep else 8, batch=2048, lr=1e-3,
    )
    return predict_torch(model, enc.transform_idx(va), scaler.transform(dense_matrix(va)))