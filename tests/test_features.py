"""Smoke-transport features must be physical and must not leak the future."""

import numpy as np
import pandas as pd

from aqf import features as fe


def _fires(lat, lon, t, frp=100.0):
    return pd.DataFrame({"time": [pd.Timestamp(t)], "lat": [lat], "lon": [lon], "frp": [frp]})


def _wind(times, u, v):
    return pd.Series(u, index=times), pd.Series(v, index=times)


def test_upwind_fire_arrives_after_travel_time():
    times = pd.date_range("2025-11-01", periods=120, freq="h")
    # Fire ~250 km NW of Delhi; a 10 m/s north-westerly blows straight at Delhi.
    lat, lon = 30.2, 75.6
    ue, un = fe.unit_vector_to(lat, lon, *fe.DELHI)
    u, v = _wind(times, 10 * ue, 10 * un)
    f = fe.fire_transport(_fires(lat, lon, "2025-11-01 08:00"), u, v)
    travel = f["travel_h"].iloc[0]
    assert 5 < travel < 9
    sai = fe.smoke_arrival_index(f, times, horizon=48)
    nowcast = sai[:, 0]
    peak = np.argmax(nowcast)
    assert abs(peak - (8 + travel)) <= 1.5


def test_downwind_fire_contributes_nothing():
    times = pd.date_range("2025-11-01", periods=72, freq="h")
    lat, lon = 30.2, 75.6
    ue, un = fe.unit_vector_to(lat, lon, *fe.DELHI)
    u, v = _wind(times, -8 * ue, -8 * un)  # wind blowing away from Delhi
    f = fe.fire_transport(_fires(lat, lon, "2025-11-01 08:00"), u, v)
    assert np.isnan(f["travel_h"].iloc[0])
    assert fe.smoke_arrival_index(f, times, horizon=24).sum() == 0


def test_sai_uses_only_published_fires():
    times = pd.date_range("2025-11-01", periods=72, freq="h")
    lat, lon = 30.2, 75.6
    ue, un = fe.unit_vector_to(lat, lon, *fe.DELHI)
    u, v = _wind(times, 6 * ue, 6 * un)
    f = fe.fire_transport(_fires(lat, lon, "2025-11-01 10:00"), u, v)
    sai = fe.smoke_arrival_index(f, times, horizon=72, latency_h=3)
    # Rows for origins before detection + latency must be exactly zero for every lead.
    assert np.all(sai[:13] == 0)
    assert sai[13].sum() > 0


def test_fire_activity_is_lagged():
    times = pd.date_range("2025-11-01", periods=10, freq="h")
    lat, lon = 30.5, 75.5
    u, v = _wind(times, 0.0, 0.0)
    f = fe.fire_transport(_fires(lat, lon, "2025-11-01 02:10"), u, v)
    act = fe.fire_activity(f, times, latency_h=3)
    assert act["fire_n_24h"].iloc[4] == 0 and act["fire_n_24h"].iloc[5] > 0


def test_calendar_diwali_flag():
    cal = fe.calendar(pd.date_range("2025-10-18", periods=24 * 5, freq="h"))
    assert cal["diwali"].max() == 1 and cal["diwali"].iloc[0] == 0


def test_inversion_sign():
    f = {"t2m": np.array([288.0]), "t850": np.array([291.0]), "u10": np.array([1.0]), "v10": np.array([0.0]),
         "u850": np.array([0.0]), "v850": np.array([0.0]), "rh2m": np.array([50.0]), "blh": np.array([100.0]),
         "prate": np.array([0.0]), "tcc": np.array([0.0])}
    w = fe.derive_weather(f)
    assert w["inversion"][0] == 3.0 and np.isclose(w["log_vent"][0], np.log1p(100.0))


def test_no_fires_gives_zero_smoke():
    from aqf.data.fires import empty_fires

    times = pd.date_range("2025-11-01", periods=48, freq="h")
    u, v = _wind(times, 5.0, -5.0)
    f = fe.fire_transport(empty_fires(), u, v)
    assert fe.smoke_arrival_index(f, times, horizon=24).sum() == 0
    assert fe.fire_activity(f, times).to_numpy().sum() == 0
