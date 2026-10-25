"""Feature engineering: weather derivatives, fire→Delhi smoke transport, calendar.

The centrepiece is the **smoke arrival index (SAI)**. Every VIIRS fire pixel
is advected towards Delhi with the 850 hPa wind over the source region at
detection time. Fires whose wind does not point at Delhi contribute nothing;
the rest arrive after ``distance / along-track wind`` hours, smeared by a
Gaussian whose width grows with travel time. Because the index for a future
hour depends only on fires that have *already* been detected, it is a
legitimate known-at-origin covariate for 1–72 h forecasts: "smoke that is
already on its way".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DELHI = (28.61, 77.21)
SOURCE_BOX = {"lat_min": 29.0, "lat_max": 32.5, "lon_min": 73.5, "lon_max": 77.5}  # Punjab + Haryana paddy belt
EARTH_R = 6371.0

# Major firework nights (national holiday calendar).
DIWALI = ["2024-11-01", "2025-10-20", "2026-11-08"]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def unit_vector_to(lat_from, lon_from, lat_to, lon_to):
    """Local east/north unit vector pointing from ``from`` to ``to`` (flat-earth approx)."""
    de = np.radians(lon_to - lon_from) * np.cos(np.radians((lat_from + lat_to) / 2))
    dn = np.radians(lat_to - lat_from)
    n = np.hypot(de, dn) + 1e-12
    return de / n, dn / n


# --------------------------------------------------------------------------- weather
def derive_weather(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Physically meaningful predictors from raw GFS fields (any common shape)."""
    ws10 = np.hypot(f["u10"], f["v10"])
    ws850 = np.hypot(f["u850"], f["v850"])
    return {
        "t2m": f["t2m"] - 273.15,
        "rh2m": f["rh2m"],
        "u10": f["u10"],
        "v10": f["v10"],
        "ws10": ws10,
        "blh": f["blh"],
        "log_vent": np.log1p(np.clip(f["blh"], 0, None) * ws10),  # ventilation coefficient (m²/s)
        "rain_mmh": np.clip(f["prate"], 0, None) * 3600.0,
        "tcc": f["tcc"],
        "inversion": f["t850"] - f["t2m"],  # >0: warm air aloft traps pollutants
        "ws850": ws850,
    }


# --------------------------------------------------------------------------- fires
def fire_transport(fires: pd.DataFrame, wind_u: pd.Series, wind_v: pd.Series, decay_km: float = 600.0) -> pd.DataFrame:
    """Attach distance, along-track wind, travel time and weight to each fire pixel.

    ``wind_u/v`` are hourly 850 hPa winds (m/s) over the source region, indexed by time.
    """
    f = fires.copy()
    f["hour"] = f["time"].dt.floor("h")
    f["dist_km"] = haversine_km(f["lat"].to_numpy(), f["lon"].to_numpy(), *DELHI)
    ue, un = unit_vector_to(f["lat"].to_numpy(), f["lon"].to_numpy(), *DELHI)
    u = wind_u.reindex(f["hour"]).to_numpy()
    v = wind_v.reindex(f["hour"]).to_numpy()
    along = u * ue + v * un  # m/s towards Delhi
    f["along_ms"] = along
    with np.errstate(divide="ignore", invalid="ignore"):
        f["travel_h"] = np.where(along > 0.5, f["dist_km"] * 1000 / np.maximum(along, 0.5) / 3600, np.nan)
    frp = np.nan_to_num(f["frp"].to_numpy(), nan=5.0).clip(0, 500)
    f["weight"] = frp * np.exp(-f["dist_km"].to_numpy() / decay_km)
    f["in_source"] = (
        f["lat"].between(SOURCE_BOX["lat_min"], SOURCE_BOX["lat_max"]) & f["lon"].between(SOURCE_BOX["lon_min"], SOURCE_BOX["lon_max"])
    )
    return f


def smoke_arrival_index(
    f: pd.DataFrame, times: pd.DatetimeIndex, horizon: int, lookback_h: int = 120, latency_h: float = 3.0
) -> np.ndarray:
    """SAI[t, h] = smoke expected to reach Delhi at ``t+h`` from fires known at ``t``.

    A detection only counts once it would have been published (``latency_h``,
    FIRMS near-real-time latency is ~3 h).

    Returns an array of shape ``(len(times), horizon + 1)``; column 0 is the
    nowcast (smoke arriving now from fires already seen).
    """
    g = f.dropna(subset=["travel_h"])
    g = g[g["travel_h"] < lookback_h]
    tau = ((g["time"] - times[0]) / pd.Timedelta(hours=1)).to_numpy()
    arrive = tau + g["travel_h"].to_numpy()
    sigma = 0.3 * g["travel_h"].to_numpy() + 6.0
    w = g["weight"].to_numpy() / (sigma * np.sqrt(2 * np.pi))
    order = np.argsort(tau)
    tau, arrive, sigma, w = tau[order], arrive[order], sigma[order], w[order]
    out = np.zeros((len(times), horizon + 1), np.float32)
    hs = np.arange(horizon + 1)
    for ti in range(len(times)):
        lo, hi = np.searchsorted(tau, [ti - lookback_h, ti - latency_h], side="right")
        if hi <= lo:
            continue
        target = ti + hs[None, :]
        z = (target - arrive[lo:hi, None]) / sigma[lo:hi, None]
        out[ti] = (w[lo:hi, None] * np.exp(-0.5 * z**2)).sum(0)
    return out


def fire_activity(f: pd.DataFrame, times: pd.DatetimeIndex, latency_h: int = 3) -> pd.DataFrame:
    """Backward-looking fire counts / FRP in the source belt over 24 h and 72 h (lagged by publication latency)."""
    src = f[f["in_source"]]
    hourly = pd.DataFrame(
        {"n": src.groupby("hour").size(), "frp": src.groupby("hour")["frp"].sum(), "w": f.groupby("hour")["weight"].sum()}
    ).reindex(times, fill_value=0.0).shift(latency_h, fill_value=0.0)
    out = pd.DataFrame(index=times)
    for win in (24, 72):
        r = hourly.rolling(win, min_periods=1).sum()
        out[f"fire_n_{win}h"] = np.log1p(r["n"])
        out[f"fire_frp_{win}h"] = np.log1p(r["frp"])
        out[f"fire_wfrp_{win}h"] = np.log1p(r["w"])
    return out


# --------------------------------------------------------------------------- calendar
def calendar(times: pd.DatetimeIndex) -> pd.DataFrame:
    ist = times + pd.Timedelta(hours=5, minutes=30)
    hod = ist.hour + ist.minute / 60
    doy = ist.dayofyear
    diwali = pd.to_datetime(DIWALI)
    days_to = np.min(np.abs((ist.normalize().values[:, None] - diwali.values[None, :]) / np.timedelta64(1, "D")), axis=1)
    return pd.DataFrame(
        {
            "hod_sin": np.sin(2 * np.pi * hod / 24),
            "hod_cos": np.cos(2 * np.pi * hod / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "weekend": (ist.dayofweek >= 5).astype(float),
            "diwali": (days_to <= 1).astype(float),
        },
        index=times,
    )
