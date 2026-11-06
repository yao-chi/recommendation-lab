"""一键复现全部实验：数据体检 → 召回 → 排序(CTR) → 评估诊断 → 图表。

用法：
    python scripts/run_all.py            # 全流程
    python scripts/run_all.py --quick    # 小样本快速验证（CI 用）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reclab import diagnostics as dg
from reclab import plots
from reclab.config import (
    CTR_USERS,
    FIG_DIR,
    K_LIST,
    MAIN_K,
    N_NEG_EVAL,
    OUT_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TAB_DIR,
)
from reclab.data import data_health, item_meta, load_dataset
from reclab.features import DENSE_COLS, SPARSE_COLS, add_features, build_ctr_pairs, prepare_sparse
from reclab.metrics import (
    auc_score,
    coverage_at_k,
    ece,
    gini_exposure,
    log_loss,
    long_tail_share,
    ndcg_at_k,
    recall_at_k,
    reliability_curve,
)
from reclab.ranking import FieldEncoder, Scaler, dense_matrix, fit_fm, fit_gbdt, fit_lr
from reclab.retrieval import (
    BPRMF,
    ItemCFRecommender,
    PopularityRecommender,
    TwoTower,
    build_index,
    rank_from_scores,
    train_bpr,
    train_two_tower,
    user_history,
)
from reclab.splits import (
    ground_truth,
    leave_one_out,
    next_item_targets,
    one_target_per_user,
    random_split,
    split_summary,
    time_split,
)

import torch


def p(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB_DIR / name, index=False, encoding="utf-8-sig")
    p(f"saved table {name}")


# --------------------------------------------------------------- 召回 ----
def build_models(train, n_users, n_items, uidx, iidx, genre_tensor, seed=RANDOM_STATE):
    models = {}

    pop = PopularityRecommender().fit(train, n_users, n_items)
    models["popularity"] = pop

    cf = ItemCFRecommender(n_neighbors=200).fit(
        train, n_users, n_items, uidx=uidx, iidx=iidx
    )
    models["itemcf"] = cf

    p("  training BPR-MF ...")
    bpr = train_bpr(train, n_users, n_items, uidx, iidx, seed=seed)
    models["bpr"] = bpr

    p("  training Two-Tower ...")
    tt = train_two_tower(
        train, n_users, n_items, uidx, iidx, genre_tensor, seed=seed
    )
    models["two-tower"] = tt

    return models


def build_hist_matrix(train, n_users, n_items, uidx, iidx):
    hist = user_history(train, uidx, iidx)
    rows, cols = [], []
    for u, h in enumerate(hist):
        rows.extend([u] * len(h))
        cols.extend(h)
    from scipy import sparse as sp

    H = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )
    deg = np.asarray(H.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return (sp.diags(1.0 / deg) @ H).tocsr()


def score_all(name, models, users, uidx, n_items, H=None, genre_tensor=None):
    m = models[name]
    if name in ("popularity", "itemcf"):
        return m.score(users, uidx, n_items)
    if name == "bpr":
        with torch.no_grad():
            ui = torch.tensor([uidx[u] for u in users], dtype=torch.long)
            return m.all_scores(ui).numpy()
    if name == "two-tower":
        with torch.no_grad():
            ui = torch.tensor([uidx[u] for u in users], dtype=torch.long)
            full_items = m.all_item_matrix(genre_tensor)
            hm = torch.from_numpy(
                np.asarray(H[ui.numpy()] @ full_items.numpy(), dtype=np.float32)
            )
            users_vec = torch.nn.functional.normalize(m.user_emb(ui) + hm, dim=-1)
            items_vec = torch.nn.functional.normalize(full_items, dim=-1)
            return ((users_vec @ items_vec.T) / torch.exp(m.log_temp)).numpy()
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-ctr", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="忽略缓存的召回分数，重新训练")
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset()
    health = data_health(ds)
    p(f"dataset: {health['n_interactions']:,} interactions, {health['n_users']} users, {health['n_items']} items")
    (REPORT_DIR / "data_health.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta, genres = item_meta(ds)
    meta.to_csv(Path(TAB_DIR) / "item_meta.csv", index=False)

    # ---------------- 主实验：时间切分（下一步预测） ----------------
    train, window, cut_ts = time_split(ds.ratings, test_frac=0.2)
    targets = next_item_targets(window)
    p(f"time split at {cut_ts.date()}: train={len(train):,} window={len(window):,} targets={len(targets):,}")
    save_table(pd.DataFrame([split_summary(train, targets, "time")]), "split_time.csv")

    uidx, iidx = build_index(ds.ratings)          # 全库索引（含测试期新物品）
    n_users, n_items = len(uidx), len(iidx)
    genre_tensor = torch.tensor(
        meta.set_index("item_id")
        .reindex(sorted(iidx, key=iidx.get))[genres]
        .values.astype(np.float32)
    )
    history = user_history(train, uidx, iidx)
    H = build_hist_matrix(train, n_users, n_items, uidx, iidx)

    gt = ground_truth(targets)
    users_eval = [u for u in sorted(gt) if u in uidx]
    gt_eval = {u: {i for i in gt[u] if i in iidx} for u in users_eval}
    gt_eval = {u: v for u, v in gt_eval.items() if v}
    p(f"evaluable users: {len(gt_eval)}")

    MODEL_NAMES = ["popularity", "itemcf", "bpr", "two-tower"]
    cache_file = OUT_DIR / "cache" / "retrieval_scores.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not args.no_cache:
        p(f"loading cached retrieval scores from {cache_file.name}")
        z = np.load(cache_file)
        scores_by_model = {n: z[n] for n in MODEL_NAMES if n in z}
        models = {}
    else:
        models = build_models(train, n_users, n_items, uidx, iidx, genre_tensor)
        scores_by_model = {}
        for name in MODEL_NAMES:
            scores_by_model[name] = score_all(
                name, models, users_eval, uidx, n_items, H, genre_tensor
            )
        np.savez_compressed(cache_file, **scores_by_model)
        p(f"cached retrieval scores -> {cache_file.name}")

    inv_items = {v: k for k, v in iidx.items()}

    rec_rows, expo_rows, recs_by_model = [], [], {}
    for name in MODEL_NAMES:
        p(f"evaluating {name} ...")
        scores = scores_by_model[name]
        ranked = rank_from_scores(scores, users_eval, uidx, history, max(K_LIST))
        item_mat = np.vectorize(lambda x: inv_items[x])(ranked)
        recs_by_model[name] = item_mat
        for k in K_LIST:
            rec_rows.append(
                {
                    "model": name,
                    "k": k,
                    "recall": recall_at_k(item_mat, gt_eval, k),
                    "ndcg": ndcg_at_k(item_mat, gt_eval, k),
                    "coverage": coverage_at_k(item_mat, k, n_items),
                }
            )
        pop = np.zeros(n_items)
        cnt = train["item_id"].value_counts()
        for it, c in cnt.items():
            if it in iidx:
                pop[iidx[it]] = c
        expo = dg.exposure_summary(ranked, pop, n_items, MAIN_K)
        expo["model"] = name
        expo_rows.append(expo)

    rec_df = pd.DataFrame(rec_rows)
    save_table(rec_df, "retrieval_metrics.csv")
    expo_df = pd.DataFrame(expo_rows)
    save_table(expo_df, "exposure.csv")

    plots.fig_data_overview(
        ds.ratings, ds.ratings.groupby("item_id").size()
    )
    plots.fig_recall_curves(rec_df)
    plots.fig_popularity_bias(expo_df)

    best = rec_df[rec_df["k"] == MAIN_K].sort_values("recall", ascending=False).iloc[0]["model"]
    p(f"best retrieval model: {best}")

    # ---------------- CTR 排序 ----------------
    funnel = {
        "recall20": float(rec_df[(rec_df.model == best) & (rec_df.k == MAIN_K)]["recall"].iloc[0]),
        "recall100": float(rec_df[(rec_df.model == best) & (rec_df.k == max(K_LIST))]["recall"].iloc[0]),
        "ndcg10_rank": float("nan"),
        "ndcg10_final": float("nan"),
    }

    if not args.skip_ctr:
        p("building CTR samples ...")
        pairs = build_ctr_pairs(train, targets, meta, n_users=CTR_USERS, n_neg=N_NEG_EVAL)
        p(f"  pairs: {len(pairs):,} (pos={int(pairs.label.sum()):,})")

        # 召回分数作为排序特征：用最优召回模型的全库分数
        score_users = users_eval
        sc = scores_by_model[best]
        retrieval_df = []
        for row, u in enumerate(score_users):
            s = sc[row]
            top = np.argsort(-s)[:500]
            for t in top:
                retrieval_df.append((u, inv_items[t], float(s[t])))
        retrieval_df = pd.DataFrame(retrieval_df, columns=["user_id", "item_id", "retrieval_score"])

        feat = add_features(pairs, train, meta, genres, ds.users, retrieval_df)
        feat = prepare_sparse(feat)
        feat = feat.dropna(subset=["label"])

        # 按用户切分，避免同一用户同时出现在训练与评估里
        rng = np.random.default_rng(RANDOM_STATE)
        users = feat["user_id"].drop_duplicates().to_numpy().copy()
        rng.shuffle(users)
        n_tr = int(len(users) * 0.7)
        tr_users, va_users = set(users[:n_tr]), set(users[n_tr:])
        tr = feat[feat["user_id"].isin(tr_users)].reset_index(drop=True)
        va = feat[feat["user_id"].isin(va_users)].reset_index(drop=True)
        p(f"  ctr train={len(tr):,} eval={len(va):,} users={len(va_users)}")

        enc = FieldEncoder().fit(pd.concat([tr, va], ignore_index=True))
        scaler = Scaler().fit(dense_matrix(tr))

        models_ctr, preds = {}, {}
        p("  fitting LR ...")
        preds["LR"] = fit_lr(enc, scaler, tr, va)
        p("  fitting GBDT ...")
        preds["GBDT"] = fit_gbdt(enc, scaler, tr, va)
        p("  fitting FM ...")
        preds["FM"] = fit_fm(enc, scaler, tr, va, deep=False)
        p("  fitting DeepFM ...")
        preds["DeepFM"] = fit_fm(enc, scaler, tr, va, deep=True)

        y = va["label"].values
        rows, roc_data, cal_data = [], {}, {}
        for name, pr in preds.items():
            auc = auc_score(y, pr)
            rows.append(
                {
                    "model": name,
                    "auc": round(auc, 4),
                    "logloss": round(log_loss(y, pr), 4),
                    "ece": round(ece(y, pr), 4),
                }
            )
            roc_data[name] = _roc(y, pr, auc)
            cal_data[name] = reliability_curve(y, pr)
        ctr_df = pd.DataFrame(rows)
        save_table(ctr_df, "ctr_metrics.csv")
        plots.fig_roc(roc_data)
        plots.fig_calibration(cal_data)
        plots.fig_ctr_metrics(ctr_df)

        # 排序后 NDCG@10：用最优 CTR 模型对候选重排
        best_ctr = ctr_df.sort_values("auc", ascending=False).iloc[0]["model"]
        va = va.assign(pred=preds[best_ctr])
        ndcg10 = _grouped_ndcg(va, k=10)
        funnel["ndcg10_rank"] = ndcg10
        funnel["ndcg10_final"] = ndcg10
        p(f"  best ctr={best_ctr} ndcg@10={ndcg10:.4f}")

    (REPORT_DIR / "funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    plots.fig_funnel(funnel)

    # ---------------- 评估诊断 ----------------
    p("running split-bias study (3 splits x BPR, same 1-target protocol) ...")
    split_rows = []
    rand_tr, rand_te = random_split(ds.ratings)
    loo_tr, _loo_va, loo_te = leave_one_out(ds.ratings)
    for split_name, (tr_, te_) in {
        "random": (rand_tr, one_target_per_user(rand_te)),
        "leave-one-out": (loo_tr, loo_te),
    }.items():
        g_ = ground_truth(te_)
        g_ = {u: {i for i in v if i in iidx} for u, v in g_.items() if u in uidx}
        g_ = {u: v for u, v in g_.items() if v}
        m_ = train_bpr(tr_, n_users, n_items, uidx, iidx, epochs=8)
        sc_ = score_all("bpr", {"bpr": m_}, list(g_), uidx, n_items)
        hist_ = user_history(tr_, uidx, iidx)
        rk_ = rank_from_scores(sc_, list(g_), uidx, hist_, MAIN_K)
        im_ = np.vectorize(lambda x: inv_items[x])(rk_)
        split_rows.append(
            {
                "split": split_name,
                "recall20": recall_at_k(im_, g_, MAIN_K),
                "n_users": len(g_),
            }
        )
        save_table(pd.DataFrame([split_summary(tr_, te_, split_name)]), f"split_{split_name}.csv")

    split_rows.append(
        {
            "split": "time",
            "recall20": float(rec_df[(rec_df.model == "bpr") & (rec_df.k == MAIN_K)]["recall"].iloc[0]),
            "n_users": len(gt_eval),
        }
    )
    split_df = pd.DataFrame(split_rows)
    save_table(split_df, "split_bias.csv")
    plots.fig_split_bias(split_df)

    p("sampling-bias study ...")
    samp_rows = []
    for name in MODEL_NAMES:
        full = float(rec_df[(rec_df.model == name) & (rec_df.k == MAIN_K)]["recall"].iloc[0])
        sc = scores_by_model[name]
        sampled = dg.sampled_candidate_recall(
            sc, users_eval, np.array([inv_items[i] for i in range(n_items)]), gt_eval, MAIN_K
        )
        samp_rows.append({"model": name, "full_catalog": full, "sampled_100": sampled})
    samp_df = pd.DataFrame(samp_rows)
    save_table(samp_df, "sampling_bias.csv")
    plots.fig_sampling_bias(samp_df)

    p("cold-start study ...")
    ubin = dg.user_activity_bins(train, bins=(0, 5, 20, 50, 10**9))
    cold_rows = []
    for name in ("popularity", "itemcf", "bpr", "two-tower"):
        if name not in recs_by_model:
            continue
        st = dg.stratified_recall(recs_by_model[name], users_eval, gt_eval, MAIN_K, ubin)
        st["model"] = name
        cold_rows.append(st)
    cold_df = pd.concat(cold_rows, ignore_index=True)
    save_table(cold_df, "cold_start.csv")
    plots.fig_cold_start(cold_df)

    # ---------------- 总结 ----------------
    summary = {
        "data": health,
        "split": {"cut_ts": str(cut_ts.date()), "train": len(train), "targets": len(targets)},
        "best_retrieval": best,
        "retrieval": {f"{r['model']}@{r['k']}": round(r["recall"], 4) for _, r in rec_df.iterrows()},
        "split_bias": split_df.set_index("split")["recall20"].round(4).to_dict(),
        "sampling_bias": samp_df.set_index("model").round(4).to_dict("index"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (REPORT_DIR / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    p(f"done in {summary['elapsed_sec']}s -> {REPORT_DIR}")


def _roc(y, p, auc):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr, auc


def _grouped_ndcg(df: pd.DataFrame, k: int) -> float:
    vals = []
    for _, g in df.groupby("user_id"):
        g = g.sort_values("pred", ascending=False).head(k)
        rel = g["label"].values
        if rel.sum() == 0:
            continue
        dcg = sum(rel[i] / np.log2(i + 2) for i in range(len(rel)))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(int(rel.sum()), k)))
        vals.append(dcg / ideal)
    return float(np.mean(vals)) if vals else float("nan")


if __name__ == "__main__":
    main()