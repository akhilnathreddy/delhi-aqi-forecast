"""Serving pieces: OpenAQ API parsing, scaler round-trip, LightGBM save/load."""

import numpy as np
import pandas as pd
import pytest

from aqf.data.openaq_api import parse_hours


def test_parse_hours_uses_period_start_in_utc():
    payload = {
        "results": [
            {"value": 120.5, "period": {"datetimeFrom": {"utc": "2026-10-24T05:00:00Z", "local": "2026-10-24T10:30:00+05:30"}}},
            {"value": None, "period": {"datetimeFrom": {"utc": "2026-10-24T06:00:00Z"}}},
            {"value": 98.0, "period": {"datetimeFrom": {"utc": "2026-10-24T04:00:00Z"}}},
        ]
    }
    s = parse_hours(payload)
    assert list(s.index) == [pd.Timestamp("2026-10-24 04:00"), pd.Timestamp("2026-10-24 05:00")]
    assert s.iloc[1] == 120.5 and s.index.tz is None


def test_lightgbm_save_load_roundtrip(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    from aqf.models.baselines import LightGBMForecaster

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"m6": rng.normal(4, 0.5, 400), "h": rng.integers(1, 73, 400).astype(float)})
    X["station"] = pd.Categorical(rng.integers(0, 3, 400))
    y = X["m6"] + 0.01 * X["h"] + rng.normal(0, 0.1, 400)
    f = LightGBMForecaster([0.1, 0.5, 0.9], residual=True)
    f.models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=20, verbose=-1).fit(X, y - X["m6"]) for q in f.quantiles}
    f.save(tmp_path)
    g = LightGBMForecaster.load(tmp_path, [0.1, 0.5, 0.9])
    for q in f.quantiles:
        np.testing.assert_allclose(f.models[q].predict(X), g.models[q].predict(X), rtol=1e-6)
