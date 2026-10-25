"""Forecast verification: point, daily-mean / AQI-category and probabilistic scores.

All functions take predictions shaped ``[origins, H, N]`` (µg/m³) and the
matching observations with NaN where a station was offline; missing targets
are excluded pairwise.
"""

from __future__ import annotations

import numpy as np

# India National AQI breakpoints for 24-h PM2.5 (µg/m³).
NAQI_BREAKS = np.array([30, 60, 90, 120, 250])
NAQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def _valid(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]


def point_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = _valid(y, p)
    if y.size == 0:
        return {"n": 0}
    err = p - y
    ss_res, ss_tot = float((err**2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "corr": float(np.corrcoef(y, p)[0, 1]) if y.size > 2 else float("nan"),
    }


def by_horizon(y: np.ndarray, p: np.ndarray) -> dict[str, list[float]]:
    """MAE and RMSE for every lead hour 1..H."""
    out = {"mae": [], "rmse": []}
    for h in range(y.shape[1]):
        s = point_scores(y[:, h], p[:, h])
        out["mae"].append(s.get("mae", np.nan))
        out["rmse"].append(s.get("rmse", np.nan))
    return out


def daily_means(y: np.ndarray, p: np.ndarray, min_hours: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """24-h block means for days 1, 2, 3 ahead → ``[origins, 3, N]`` each."""
    days = y.shape[1] // 24
    yd = np.full((y.shape[0], days, y.shape[2]), np.nan)
    pd_ = np.full_like(yd, np.nan)
    for d in range(days):
        ys, ps = y[:, 24 * d : 24 * (d + 1)], p[:, 24 * d : 24 * (d + 1)]
        ok = np.isfinite(ys)
        cnt = ok.sum(1)
        with np.errstate(invalid="ignore"):
            yd[:, d] = np.where(cnt >= min_hours, np.nansum(ys, 1) / np.maximum(cnt, 1), np.nan)
            pd_[:, d] = np.where(cnt >= min_hours, np.nansum(np.where(ok, ps, 0), 1) / np.maximum(cnt, 1), np.nan)
    return yd, pd_


def naqi_category(x: np.ndarray) -> np.ndarray:
    return np.digitize(x, NAQI_BREAKS, right=True)


def category_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = _valid(y, p)
    cy, cp = naqi_category(y), naqi_category(p)
    severe_obs, severe_fc = y > 250, p > 250
    hits = int((severe_obs & severe_fc).sum())
    return {
        "category_acc": float((cy == cp).mean()),
        "category_within1": float((np.abs(cy - cp) <= 1).mean()),
        # Exceedance skill for "Very Poor or worse" (>120 µg/m³) days.
        "vp_pod": float(((y > 120) & (p > 120)).sum() / max((y > 120).sum(), 1)),
        "vp_far": float(((y <= 120) & (p > 120)).sum() / max((p > 120).sum(), 1)),
        "severe_pod": hits / max(int(severe_obs.sum()), 1),
        "severe_obs": int(severe_obs.sum()),
    }


def pinball(y: np.ndarray, q_pred: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    """Mean pinball loss per quantile and central-interval coverage."""
    out = {}
    m = np.isfinite(y)
    for k, q in enumerate(quantiles):
        d = y[m] - q_pred[..., k][m]
        out[f"pinball_q{int(q * 100)}"] = float(np.mean(np.maximum(q * d, (q - 1) * d)))
    lo, hi = q_pred[..., 0][m], q_pred[..., -1][m]
    out["coverage"] = float(((y[m] >= lo) & (y[m] <= hi)).mean())
    out["nominal_coverage"] = quantiles[-1] - quantiles[0]
    out["mean_interval_width"] = float((hi - lo).mean())
    return out


def evaluate(y: np.ndarray, q_pred: np.ndarray, quantiles: list[float], report_h=(24, 48, 72)) -> dict:
    """Full scorecard. ``q_pred`` is ``[origins, H, N, Q]``; the median is the point forecast."""
    med = q_pred[..., quantiles.index(0.5)]
    res = {"overall": point_scores(y, med), "by_horizon": by_horizon(y, med)}
    for h in report_h:
        res[f"h{h}"] = point_scores(y[:, h - 1], med[:, h - 1])
    yd, pdm = daily_means(y, med)
    for d in range(yd.shape[1]):
        res[f"day{d + 1}"] = point_scores(yd[:, d], pdm[:, d]) | category_scores(yd[:, d], pdm[:, d])
    if q_pred.shape[-1] > 1:
        res["probabilistic"] = pinball(y, q_pred, quantiles)
    return res
