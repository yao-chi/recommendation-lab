"""切分逻辑测试：三种切分不能互相泄漏。"""

import pandas as pd
import pytest

from reclab.splits import ground_truth, leave_one_out, random_split, time_split


@pytest.fixture
def toy():
    rows = []
    for u in [1, 2, 3]:
        for k in range(25):
            rows.append(
                {
                    "user_id": u,
                    "item_id": 100 + k,
                    "rating": 4,
                    "ts": pd.Timestamp("2020-01-01") + pd.Timedelta(days=k),
                }
            )
    return pd.DataFrame(rows)


def test_leave_one_out_keeps_last_per_user(toy):
    train, val, test = leave_one_out(toy, min_interactions=20)
    assert len(test) == 3
    for u in [1, 2, 3]:
        assert test[test.user_id == u]["ts"].max() == toy[toy.user_id == u]["ts"].max()
        # 训练集里不能出现测试物品
        assert not set(test[test.user_id == u].item_id) & set(train[train.user_id == u].item_id)


def test_time_split_is_temporal(toy):
    train, test, cut = time_split(toy, test_frac=0.1)
    assert train["ts"].max() < cut
    assert test["ts"].min() >= cut


def test_random_split_disjoint(toy):
    train, test = random_split(toy, test_frac=0.2)
    assert len(set(train.index) & set(test.index)) == 0


def test_ground_truth_shape(toy):
    _, _, test = leave_one_out(toy, min_interactions=20)
    gt = ground_truth(test)
    assert set(gt) == {1, 2, 3}
    assert all(len(v) == 1 for v in gt.values())