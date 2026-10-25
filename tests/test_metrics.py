import numpy as np

from aqf import metrics as M


def test_perfect_forecast():
    y = np.random.default_rng(0).uniform(20, 300, (5, 72, 4))
    q = np.stack([y * 0.8, y, y * 1.2], -1)
    r = M.evaluate(y, q, [0.1, 0.5, 0.9])
    assert r["overall"]["mae"] == 0 and r["day1"]["category_acc"] == 1
    assert r["probabilistic"]["coverage"] == 1


def test_missing_targets_are_ignored():
    y = np.full((2, 72, 3), 100.0)
    y[0, :10] = np.nan
    p = np.full_like(y, 110.0)
    s = M.point_scores(y, p)
    assert s["mae"] == 10 and s["n"] == 2 * 72 * 3 - 30


def test_naqi_categories():
    assert list(M.naqi_category(np.array([10, 30, 31, 95, 200, 400]))) == [0, 0, 1, 3, 4, 5]


def test_daily_means_require_enough_hours():
    y = np.full((1, 72, 1), 50.0)
    y[0, 24:40] = np.nan  # day 2 has only 8 valid hours
    yd, _ = M.daily_means(y, y)
    assert yd[0, 0, 0] == 50 and np.isnan(yd[0, 1, 0])
