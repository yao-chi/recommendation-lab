"""指标函数的单元测试：指标写错，后面所有结论都白搭。"""

import numpy as np
import pytest

from reclab.metrics import (
    auc_score,
    coverage_at_k,
    ece,
    gini_exposure,
    log_loss,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_perfect_and_zero():
    gt = {1: {10, 11}, 2: {20}}
    recs = np.array([[10, 11, 0], [20, 0, 0]])
    assert recall_at_k(recs, gt, 2) == pytest.approx(1.0)
    bad = np.array([[99, 98, 97], [96, 95, 94]])
    assert recall_at_k(bad, gt, 2) == 0.0


def test_ndcg_rewards_early_hits():
    gt = {1: {10}}
    early = np.array([[10, 5, 6]])
    late = np.array([[5, 6, 10]])
    assert ndcg_at_k(early, gt, 3) > ndcg_at_k(late, gt, 3)


def test_ndcg_perfect_equals_one():
    gt = {1: {1, 2}, 2: {3, 4}}
    recs = np.array([[1, 2], [3, 4]])
    assert ndcg_at_k(recs, gt, 2) == pytest.approx(1.0)


def test_coverage_counts_distinct_items():
    recs = np.array([[1, 1], [1, 1]])
    assert coverage_at_k(recs, 2, 10) == pytest.approx(0.1)


def test_gini_zero_when_uniform():
    recs = np.array([[0, 1], [2, 3]])
    assert gini_exposure(recs, 2) < 0.2


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert auc_score(y, p) == pytest.approx(0.75)


def test_log_loss_perfect_prediction_is_small():
    y = np.array([0, 1])
    assert log_loss(y, np.array([1e-9, 1 - 1e-9])) < 1e-6


def test_ece_zero_when_perfectly_calibrated_groups():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece(y, p, n_bins=5) == pytest.approx(0.0, abs=1e-6)