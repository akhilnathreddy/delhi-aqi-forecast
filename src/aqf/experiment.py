"""End-to-end experiment: fit every model on a fold, forecast the test set, score it.

Outputs
-------
``reports/metrics_{fold}.json``      scorecards for every model
``reports/importance_{fold}.json``   LightGBM gain + TFT variable-selection weights
``reports/history_{fold}.json``      neural training curves
``data/predictions/{fold}_{model}.npz``  quantile forecasts ``[origins, H, N, Q]``
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch

from .config import Config
from .dataset import ForecastData
from .metrics import evaluate
from .models import baselines as bl

log = logging.getLogger(__name__)


def fold_split(cfg: Config, d: ForecastData, fold: str, train_every_h: int | None = None):
    s = cfg["split"]
    return d.split(s["folds"][fold], s["gap_h"], cfg["forecast"]["origin_every_h"], train_every_h)


def _save_pred(cfg: Config, fold: str, name: str, origins: np.ndarray, q: np.ndarray):
    np.savez_compressed(cfg.path("predictions", f"{fold}_{name}.npz"), origins=origins, q=q.astype(np.float32))


def _load_pred(cfg: Config, fold: str, name: str):
    f = cfg.data_dir / "predictions" / f"{fold}_{name}.npz"
    return np.load(f)["q"] if f.exists() else None


def run(cfg: Config, models: list[str], fold: str = "main", epochs: int | None = None):
    torch.set_num_threads(4)
    fc = cfg["forecast"]
    quantiles = cfg["quantiles"]
    d = ForecastData(cfg.path("processed", "dataset.npz"), fc["horizon"], fc["encoder_length"], cfg["gfs"]["availability_lag_h"])
    sp = fold_split(cfg, d, fold)
    d.fit_scalers(sp.train)
    log.info("fold %s: %d train, %d val, %d test origins; %d stations", fold, len(sp.train), len(sp.val), len(sp.test), d.N)
    test = sp.test
    y_test = np.stack([d.target(t) for t in test])

    rep = cfg.reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    mpath = rep / f"metrics_{fold}.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    ipath = rep / f"importance_{fold}.json"
    importance = json.loads(ipath.read_text()) if ipath.exists() else {}
    hpath = rep / f"history_{fold}.json"
    history = json.loads(hpath.read_text()) if hpath.exists() else {}
    meta = {
        "fold": fold,
        "n_stations": int(d.N),
        "stations": [str(x) for x in d.station_name],
        "partitions": cfg["split"]["folds"][fold],
        "n_train_origins": int(len(sp.train)),
        "n_test_origins": int(len(test)),
        "test_obs_fraction": float(np.isfinite(y_test).mean()),
    }

    def score(name: str, q: np.ndarray, extra: dict | None = None):
        _save_pred(cfg, fold, name, test, q)
        qs = quantiles if q.shape[-1] == len(quantiles) else [0.5]
        metrics[name] = evaluate(y_test, q, qs, tuple(fc["report_horizons"])) | (extra or {})
        o = metrics[name]
        log.info("%-22s MAE %.1f | h24 %.1f h48 %.1f h72 %.1f", name, o["overall"]["mae"], o["h24"]["mae"], o["h48"]["mae"], o["h72"]["mae"])
        mpath.write_text(json.dumps({"_meta": meta} | {k: v for k, v in metrics.items() if k != "_meta"}, indent=1))

    def pad_q(p):  # deterministic baselines: replicate the point forecast for every quantile slot
        return np.repeat(p, len(quantiles), -1)

    for name in models:
        t0 = time.time()
        if name == "persistence":
            score(name, pad_q(bl.persistence(d, test)))
        elif name == "diurnal_persistence":
            score(name, pad_q(bl.diurnal_persistence(d, test)))
        elif name == "climatology":
            score(name, pad_q(bl.Climatology().fit(d, sp.train).predict(d, test)))
        elif name.startswith("lightgbm"):
            drop = {
                "lightgbm": (),
                "lightgbm_resid": (),
                "lightgbm_nofire": ("fire_", "sai"),
                "lightgbm_resid_nofire": ("fire_", "sai"),
                "lightgbm_nofcwx": ("fc_", "fcmean_"),
            }[name]
            m = bl.LightGBMForecaster(quantiles, drop=drop, seed=cfg["seed"], residual="resid" in name).fit(d, sp.train, sp.val)
            score(name, m.predict(d, test), {"train_s": time.time() - t0})
            imp = m.importance()
            importance[name] = (imp / imp.sum()).head(40).round(5).to_dict()
        elif name in ("tft", "stgnn", "tft_anchored", "stgnn_anchored"):
            from .train import fit, predict, tft_interpretation

            is_tft = name.startswith("tft")
            n_ep = epochs or (15 if is_tft else 20)
            # The graph model sees one sample per origin, so it uses 3-hourly origins.
            tr = sp.train if is_tft else fold_split(cfg, d, fold, train_every_h=3).train
            model, hist = fit(
                name, d, tr, sp.val, quantiles, epochs=n_ep, seed=cfg["seed"], time_budget_s=80 * 60,
                origins_per_batch=8,
                stations_per_origin=24 if is_tft else None,
            )
            history[name] = hist
            q = predict(name, model, d, test)
            score(name, q, {"train_s": time.time() - t0, "epochs": len(hist)})
            torch.save(model.state_dict(), cfg.path("models", f"{fold}_{name}.pt"))
            if is_tft:
                importance[name] = tft_interpretation(model, d, test[:: max(1, len(test) // 60)])
        elif name == "ensemble":
            members = ("lightgbm", "tft", "stgnn")
            parts = [p for p in (_load_pred(cfg, fold, k) for k in members) if p is not None]
            if len(parts) >= 2:
                q = np.expm1(np.mean([np.log1p(np.clip(p, 0, None)) for p in parts], 0))
                score(name, np.sort(q, -1), {"members": len(parts)})
        elif name == "ensemble_anchored":
            members = ("lightgbm_resid", "tft_anchored", "stgnn_anchored")
            parts = [p for p in (_load_pred(cfg, fold, k) for k in members) if p is not None]
            if len(parts) >= 2:
                q = np.expm1(np.mean([np.log1p(np.clip(p, 0, None)) for p in parts], 0))
                score(name, np.sort(q, -1), {"members": len(parts)})
        else:
            raise ValueError(name)
        ipath.write_text(json.dumps(importance, indent=1))
        hpath.write_text(json.dumps(history, indent=1))

    # Skill relative to persistence (1 - MAE / MAE_persistence) for headline horizons.
    if "persistence" in metrics:
        p = metrics["persistence"]
        for k, v in metrics.items():
            if k.startswith("_") or not isinstance(v, dict) or "h24" not in v:
                continue
            v["skill_vs_persistence"] = {
                h: 1 - v[h]["mae"] / p[h]["mae"] for h in ("h24", "h48", "h72", "day1", "day2", "day3") if h in v and h in p
            }
        mpath.write_text(json.dumps({"_meta": meta} | {k: v for k, v in metrics.items() if k != "_meta"}, indent=1))
    return metrics
