"""Training loops for the neural forecasters (CPU-friendly, early-stopped)."""

from __future__ import annotations

import copy
import logging
import time

import numpy as np
import torch

from .dataset import ForecastData
from .models.stgnn import WindAwareSTGNN
from .models.tft import TemporalFusionTransformer

log = logging.getLogger(__name__)

# Uses a GPU when one is present (e.g. the Colab notebook), otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """Pinball loss averaged over quantiles and observed targets (NaN = missing)."""
    mask = torch.isfinite(target)
    t = torch.nan_to_num(target)
    q = torch.tensor(quantiles, device=pred.device)
    d = t.unsqueeze(-1) - pred
    loss = torch.maximum(q * d, (q - 1) * d).mean(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


# --------------------------------------------------------------------------- batching
def _wind_idx(d: ForecastData) -> tuple[int, int]:
    return d.wx_names.index("u10"), d.wx_names.index("v10")


def graph_batch(d: ForecastData, origins: np.ndarray):
    iu, iv = _wind_idx(d)
    past = np.stack([d.past(t) for t in origins])
    fut = np.stack([d.future(t) for t in origins])
    y = np.stack([d.scale_y(d.target(t)) for t in origins])
    pw = np.stack([d.wx_an[t - d.L + 1 : t + 1][:, :, [iu, iv]].mean(1) for t in origins])
    fw = np.stack([np.nan_to_num(d.wx_fc[d.cycle_for[t], d.lead0[t] + 1 : d.lead0[t] + 1 + d.H][:, :, [iu, iv]]).mean(1) for t in origins])
    return [torch.from_numpy(a.astype(np.float32)) for a in (past, fut, y, pw, fw)]


def station_batch(d: ForecastData, origins: np.ndarray, stations: np.ndarray | None = None):
    """Flatten (origin, station) pairs for the per-series TFT.

    ``stations`` optionally restricts each origin to a subset of stations
    (random per batch during training, which cuts epoch time without
    dropping any origin).
    """
    past, fut, y, _, _ = graph_batch(d, origins)
    st_idx = torch.arange(d.N) if stations is None else torch.as_tensor(stations)
    past, fut, y = past[:, :, st_idx], fut[:, :, st_idx], y[:, :, st_idx]
    B, L, N, Fp = past.shape
    past = past.permute(0, 2, 1, 3).reshape(B * N, L, Fp)
    fut = fut.permute(0, 2, 1, 3).reshape(B * N, d.H, -1)
    y = y.permute(0, 2, 1).reshape(B * N, d.H)
    st = st_idx.repeat(B)
    net = torch.from_numpy(d.net).long()[st_idx].repeat(B)
    ll = torch.from_numpy(np.stack([(d.lat - 28.6) / 0.3, (d.lon - 77.2) / 0.3], -1).astype(np.float32))[st_idx].repeat(B, 1)
    return past, fut, y, st, net, ll


# --------------------------------------------------------------------------- models
def base_name(name: str) -> str:
    return name.removesuffix("_anchored")


def build_model(name: str, d: ForecastData, quantiles: list[float]):
    n_past, n_fut = len(d.past_names), len(d.future_names)
    name = base_name(name)
    if name == "tft":
        return TemporalFusionTransformer(n_past, n_fut, d.N, 8, d.H, quantiles, d=48, heads=4, dropout=0.1)
    if name == "stgnn":
        return WindAwareSTGNN(n_past, n_fut, d.lat, d.lon, d.H, quantiles, d=64)
    raise ValueError(name)


def _forward(name, model, d, origins, stations=None):
    """``*_anchored`` models predict the offset from the last-6-h mean (scaled log PM2.5)."""
    anchored = name.endswith("_anchored")
    if base_name(name) == "tft":
        past, fut, y, st, net, ll = (t.to(DEVICE) for t in station_batch(d, origins, stations))
        pred = model(past, fut, st, net, ll)  # [B*N, H, Q]
        if anchored:
            pred = pred + past[:, -6:, 0].mean(1)[:, None, None]
        return pred, y
    past, fut, y, pw, fw = (t.to(DEVICE) for t in graph_batch(d, origins))
    pred = model(past, fut, pw, fw)  # [B, H, N, Q]
    if anchored:
        pred = pred + past[:, -6:, :, 0].mean(1)[:, None, :, None]
    return pred, y


def fit(
    name: str,
    d: ForecastData,
    train_origins: np.ndarray,
    val_origins: np.ndarray,
    quantiles: list[float],
    epochs: int = 20,
    origins_per_batch: int = 4,
    lr: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
    time_budget_s: float = 3 * 3600,
    stations_per_origin: int | None = None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name, d, quantiles).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("%s: %d parameters, %d train / %d val origins", name, n_params, len(train_origins), len(val_origins))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_origins)
        losses = []
        for s in range(0, len(perm), origins_per_batch):
            sub = rng.choice(d.N, stations_per_origin, replace=False) if stations_per_origin else None
            pred, y = _forward(name, model, d, perm[s : s + origins_per_batch], sub)
            loss = quantile_loss(pred, y, quantiles)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val = evaluate_loss(name, model, d, val_origins, quantiles)
        sched.step(val)
        history.append({"epoch": ep + 1, "train": float(np.mean(losses)), "val": val, "elapsed_s": time.time() - t0})
        log.info("%s epoch %d train %.4f val %.4f (%.0fs)", name, ep + 1, np.mean(losses), val, time.time() - t0)
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if time.time() - t0 > time_budget_s:
            log.info("%s: time budget reached", name)
            break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_loss(name, model, d, origins, quantiles, origins_per_batch: int = 8) -> float:
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(origins), origins_per_batch):
        pred, y = _forward(name, model, d, origins[s : s + origins_per_batch])
        m = torch.isfinite(y).sum().item()
        tot += quantile_loss(pred, y, quantiles).item() * m
        n += m
    return tot / max(n, 1)


@torch.no_grad()
def predict(name, model, d: ForecastData, origins: np.ndarray, origins_per_batch: int = 8) -> np.ndarray:
    """→ ``[origins, H, N, Q]`` in µg/m³."""
    model.eval()
    out = []
    for s in range(0, len(origins), origins_per_batch):
        o = origins[s : s + origins_per_batch]
        pred, _ = _forward(name, model, d, o)
        if base_name(name) == "tft":
            pred = pred.view(len(o), d.N, d.H, -1).permute(0, 2, 1, 3)
        out.append(d.unscale_y(pred.cpu().numpy()))
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def tft_interpretation(model, d: ForecastData, origins: np.ndarray, origins_per_batch: int = 8) -> dict:
    """Mean variable-selection weights (past / future) and attention by lag."""
    model.eval()
    wp, wf, att = [], [], []
    for s in range(0, len(origins), origins_per_batch):
        past, fut, _, st, net, ll = (t.to(DEVICE) for t in station_batch(d, origins[s : s + origins_per_batch]))
        _, w = model(past, fut, st, net, ll, return_weights=True)
        wp.append(w["past"].mean((0, 1)).cpu().numpy())
        wf.append(w["future"].mean((0, 1)).cpu().numpy())
        att.append(w["attention"].mean(0).cpu().numpy())
    return {
        "past_importance": dict(zip(d.past_names, np.mean(wp, 0).tolist())),
        "future_importance": dict(zip(d.future_names, np.mean(wf, 0).tolist())),
        "attention": np.mean(att, 0).tolist(),  # [H, L+H]
    }
