"""Production model bundle: fit on all data, save it, load it and forecast.

The bundle in ``models/final/`` is what the Streamlit app ships with:

``meta.json``                 stations (order matters), scalers, feature names, config
``lightgbm_q{10,50,90}.txt``  level-anchored LightGBM boosters
``tft_anchored.pt``           level-anchored Temporal Fusion Transformer weights
``stgnn_anchored.pt``         level-anchored wind-aware ST-GNN weights

The level-anchored models are used because they were the only ones that
held up on the held-out winter season, and they cost little in other months.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ROOT, Config
from .dataset import ForecastData
from .models import baselines as bl

log = logging.getLogger(__name__)

BUNDLE_DIR = ROOT / "models" / "final"
NEURAL = ("tft_anchored", "stgnn_anchored")


def train_final(cfg: Config, models: tuple[str, ...] = ("lightgbm_resid", *NEURAL), out_dir: Path = BUNDLE_DIR):
    """Fit the serving models on the ``final`` fold (all data; October 2026 for early stopping)."""
    torch.set_num_threads(4)
    fc, quantiles = cfg["forecast"], cfg["quantiles"]
    d = ForecastData(cfg.path("processed", "dataset.npz"), fc["horizon"], fc["encoder_length"], cfg["gfs"]["availability_lag_h"])
    fold = cfg["split"]["folds"]["final"]
    sp = d.split(fold, cfg["split"]["gap_h"], fc["origin_every_h"])
    d.fit_scalers(sp.train)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "stations": [
                {"location_id": int(i), "name": str(n), "lat": float(la), "lon": float(lo), "network": int(nw)}
                for i, n, la, lo, nw in zip(d.station_id, d.station_name, d.lat, d.lon, d.net)
            ],
            "scalers": d.scaler_state(),
            "past_names": d.past_names,
            "future_names": d.future_names,
            "quantiles": quantiles,
            "horizon": d.H,
            "encoder_length": d.L,
            "availability_lag_h": d.lag_h,
            "trained_on": fold,
            "models": meta.get("models", {}),
        }
    )
    for name in models:
        t0 = time.time()
        if name == "lightgbm_resid":
            m = bl.LightGBMForecaster(quantiles, seed=cfg["seed"], residual=True).fit(d, sp.train, sp.val)
            meta["models"][name] = {"files": m.save(out_dir), "train_s": time.time() - t0}
        else:
            from .train import fit

            is_tft = name.startswith("tft")
            tr = sp.train if is_tft else d.split(fold, cfg["split"]["gap_h"], 3).train
            model, hist = fit(
                name, d, tr, sp.val, quantiles, epochs=15 if is_tft else 20, seed=cfg["seed"], time_budget_s=80 * 60,
                origins_per_batch=8, stations_per_origin=24 if is_tft else None,
            )
            torch.save(model.state_dict(), out_dir / f"{name}.pt")
            meta["models"][name] = {"file": f"{name}.pt", "epochs": len(hist), "best_val": min(h["val"] for h in hist), "train_s": time.time() - t0}
        meta_path.write_text(json.dumps(meta, indent=1))
        log.info("saved %s to %s", name, out_dir)
    return meta


class Bundle:
    """Loaded serving models plus everything needed to rebuild their inputs."""

    def __init__(self, path: Path = BUNDLE_DIR):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.quantiles = self.meta["quantiles"]
        self.stations = pd.DataFrame(self.meta["stations"])

    def available(self) -> list[str]:
        return [k for k in self.meta["models"] if (self.path / self.meta["models"][k].get("file", "meta.json")).exists()]

    def predict(self, d: ForecastData, origins: np.ndarray) -> dict[str, np.ndarray]:
        """Quantile forecasts ``[origins, H, N, Q]`` (µg/m³) from every bundled model + their ensemble."""
        from .train import DEVICE, build_model
        from .train import predict as nn_predict

        d.load_scaler_state(self.meta["scalers"])
        out = {}
        if "lightgbm_resid" in self.meta["models"]:
            m = bl.LightGBMForecaster.load(self.path, self.quantiles, residual=True)
            out["lightgbm_resid"] = m.predict(d, origins)
        for name in NEURAL:
            if name not in self.meta["models"]:
                continue
            model = build_model(name, d, self.quantiles).to(DEVICE)
            model.load_state_dict(torch.load(self.path / f"{name}.pt", map_location=DEVICE))
            out[name] = nn_predict(name, model, d, origins)
        if len(out) >= 2:
            ens = np.expm1(np.mean([np.log1p(np.clip(p, 0, None)) for p in out.values()], 0))
            out["ensemble_anchored"] = np.sort(ens, -1).astype(np.float32)
        return out


def add_sensor_ids(out_dir: Path = BUNDLE_DIR, year: int | None = None):
    """Store each station's OpenAQ sensor ids (from the archive) so the app can use the v3 API."""
    import concurrent.futures as cf

    from .data.openaq import sensor_ids

    meta_path = Path(out_dir) / "meta.json"
    meta = json.loads(meta_path.read_text())
    year = year or pd.Timestamp.now().year
    ids = [s["location_id"] for s in meta["stations"]]
    with cf.ThreadPoolExecutor(16) as ex:
        found = dict(zip(ids, ex.map(lambda i: sensor_ids(i, year), ids)))
    keep = ("pm25", "pm10", "no2", "co", "o3")
    meta["sensors"] = {str(i): {p: s for p, s in v.items() if p in keep} for i, v in found.items()}
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta["sensors"]


def export_backtest(cfg: Config, out_path: Path = ROOT / "app" / "backtest.npz", every_h: int = 12):
    """Compact test-period forecasts for the app's backtest explorer (float16, 12-hourly origins)."""
    fc = cfg["forecast"]
    d = ForecastData(cfg.path("processed", "dataset.npz"), fc["horizon"], fc["encoder_length"], cfg["gfs"]["availability_lag_h"])
    arrays = {
        "station_id": d.station_id,
        "station_name": d.station_name.astype(str),
        "lat": d.lat,
        "lon": d.lon,
    }
    for fold in ("main", "winter"):
        for model in ("persistence", "ensemble", "ensemble_anchored"):
            z = np.load(cfg.data_dir / "predictions" / f"{fold}_{model}.npz")
            origins, q = z["origins"], z["q"]
            sel = d.times[origins].hour % every_h == 0
            arrays[f"{fold}_{model}"] = q[sel][..., 1] if model == "persistence" else q[sel]
            arrays[f"{fold}_{model}"] = arrays[f"{fold}_{model}"].astype(np.float16)
        o = origins[sel]
        arrays[f"{fold}_origins"] = d.times[o].values.astype("datetime64[h]").astype(np.int64)
        lo, hi = int(o.min()) - 7 * 24, int(o.max()) + d.H + 1
        arrays[f"{fold}_obs_start"] = np.array(d.times[lo].value // 3_600_000_000_000)
        arrays[f"{fold}_obs"] = d.y[lo:hi].astype(np.float16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path
