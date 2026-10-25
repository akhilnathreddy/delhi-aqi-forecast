"""Figures and a markdown results summary from the saved metrics/predictions."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .data.fires import load_fires  # noqa: E402
from .dataset import ForecastData  # noqa: E402

log = logging.getLogger(__name__)

# Validated categorical palette (fixed order; colour follows the model, never its rank).
COLORS = {
    "persistence": "#9b9a96",
    "diurnal_persistence": "#c3c2b7",
    "climatology": "#52514e",
    "lightgbm": "#2a78d6",
    "tft": "#eb6834",
    "stgnn": "#1baf7a",
    "ensemble": "#6250d6",
    "lightgbm_nofire": "#e87ba4",
    "lightgbm_nofcwx": "#eda100",
    "lightgbm_resid": "#008300",
    "lightgbm_resid_nofire": "#e34948",
    "tft_anchored": "#d95926",
    "stgnn_anchored": "#199e70",
    "ensemble_anchored": "#e34948",
}
LABELS = {
    "persistence": "Persistence",
    "diurnal_persistence": "Diurnal persistence",
    "climatology": "Climatology",
    "lightgbm": "LightGBM",
    "lightgbm_nofire": "LightGBM − fire features",
    "lightgbm_nofcwx": "LightGBM − GFS forecast",
    "tft": "Temporal Fusion Transformer",
    "stgnn": "Wind-aware ST-GNN",
    "ensemble": "Ensemble (LGBM+TFT+GNN)",
    "lightgbm_resid": "LightGBM (level-anchored)",
    "lightgbm_resid_nofire": "LightGBM (level-anchored) − fire",
    "tft_anchored": "TFT (level-anchored)",
    "stgnn_anchored": "ST-GNN (level-anchored)",
    "ensemble_anchored": "Ensemble (level-anchored)",
}
ORDER = list(LABELS)
# Curated subset for line/bar charts (≤ 7 series stays legible).
PLOT_SET = ["persistence", "diurnal_persistence", "lightgbm", "tft", "stgnn", "ensemble", "ensemble_anchored"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")


def _load(cfg: Config, fold: str):
    p = cfg.reports_dir / f"metrics_{fold}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_horizon(cfg, fold, m, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    models = [k for k in PLOT_SET if k in m]
    for k in models:
        mae = m[k]["by_horizon"]["mae"]
        ax.plot(np.arange(1, len(mae) + 1), mae, color=COLORS[k], lw=2, label=LABELS[k])
    ax.set_xlabel("Lead time (hours)", color=INK2)
    ax.set_ylabel("MAE (µg/m³)", color=INK2)
    ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    _style(ax, f"Hourly PM2.5 error by lead time — {fold} test")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_skill(cfg, fold, m, out):
    models = [k for k in PLOT_SET if k in m and k != "persistence"]
    days = ["day1", "day2", "day3"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(models)
    for i, k in enumerate(models):
        vals = [m[k].get("skill_vs_persistence", {}).get(d, np.nan) * 100 for d in days]
        ax.bar(np.arange(3) + i * w, vals, w * 0.92, color=COLORS[k], label=LABELS[k])
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(np.arange(3) + w * (len(models) - 1) / 2)
    ax.set_xticklabels(["Day 1 (1–24 h)", "Day 2 (25–48 h)", "Day 3 (49–72 h)"])
    ax.set_ylabel("MAE skill vs persistence (%)", color=INK2)
    _style(ax, f"Daily-mean PM2.5 skill over persistence — {fold} test")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_overview(cfg, d: ForecastData, out):
    """Two stacked panels sharing time: network PM2.5 and source-belt fire counts (no dual axis)."""
    daily = pd.Series(np.nanmean(d.y, 1), index=d.times).resample("D").mean()
    fd = pd.DataFrame(d.fires_daily, columns=["t", "n", "frp"])
    fd.index = pd.to_datetime(fd["t"].astype("int64").astype("datetime64[h]"))
    fd = fd["n"].reindex(daily.index, fill_value=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(daily.index, daily.values, color="#2a78d6", lw=1.6)
    for lo, hi, lab in [(0, 30, "Good"), (90, 120, "Poor"), (250, 2000, "Severe")]:
        a1.axhspan(lo, hi, color="#f1f0ec", zorder=0)
        a1.text(daily.index[-1], min(hi, 315) - 4, f"NAQI {lab}", ha="right", va="top", fontsize=8, color=INK2)
    a1.set_ylabel("PM2.5 (µg/m³)", color=INK2)
    a1.set_ylim(0, max(320, np.nanmax(daily.values) * 1.05))
    _style(a1, "Network-mean daily PM2.5 across Delhi-NCR")
    a2.bar(fd.index, fd.values, width=1.0, color="#eb6834")
    a2.set_ylabel("Fire pixels / day", color=INK2)
    _style(a2, "VIIRS fire detections in the Punjab–Haryana belt (S-NPP + NOAA-20)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_map(cfg, d: ForecastData, out):
    fires = load_fires(cfg.data_dir / "raw" / "fires")
    season = fires[(fires["time"] >= "2025-10-01") & (fires["time"] < "2025-12-01")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    hb = a1.hexbin(season["lon"], season["lat"], gridsize=60, bins="log", cmap="Oranges", mincnt=1,
                   extent=(70, 80, 26, 33))
    a1.scatter([77.21], [28.61], marker="*", s=180, color="#2a78d6", edgecolor="white", zorder=3)
    a1.annotate("Delhi", (77.21, 28.61), xytext=(6, -12), textcoords="offset points", color=INK, fontsize=9)
    a1.set_xlabel("Longitude", color=INK2)
    a1.set_ylabel("Latitude", color=INK2)
    fig.colorbar(hb, ax=a1, label="fire pixels (log)")
    _style(a1, f"Stubble fires Oct–Nov 2025 ({len(season):,} VIIRS pixels)")
    a1.grid(False)
    mean = np.nanmean(d.y, 0)
    sc = a2.scatter(d.lon, d.lat, c=mean, cmap="Blues", s=40, edgecolor=INK2, lw=0.5, vmin=np.nanmin(mean), vmax=np.nanmax(mean))
    fig.colorbar(sc, ax=a2, label="mean PM2.5 (µg/m³)")
    a2.set_xlabel("Longitude", color=INK2)
    _style(a2, f"{d.N} monitoring stations")
    a2.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_importance(cfg, fold, imp, out):
    model = "tft_anchored" if "tft_anchored" in imp else "tft"
    if model not in imp:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, part, title in [(axes[0], "past_importance", "Encoder (past) inputs"), (axes[1], "future_importance", "Decoder (known-future) inputs")]:
        s = pd.Series(imp[model][part]).sort_values().tail(12)
        colors = ["#eb6834" if ("fire" in n or "sai" in n) else "#2a78d6" for n in s.index]
        ax.barh(s.index, s.values * 100, color=colors, height=0.7)
        ax.set_xlabel("mean selection weight (%)", color=INK2)
        _style(ax, title)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"TFT variable-selection weights ({fold} test) — orange = fire/smoke inputs", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_fire_ablation(cfg, fold, d: ForecastData, out):
    """Monthly MAE of LightGBM with vs without fire inputs (day-2 daily mean)."""
    from .metrics import daily_means

    base = "lightgbm_resid" if (cfg.data_dir / "predictions" / f"{fold}_lightgbm_resid_nofire.npz").exists() else "lightgbm"
    f_full = cfg.data_dir / "predictions" / f"{fold}_{base}.npz"
    f_nof = cfg.data_dir / "predictions" / f"{fold}_{base}_nofire.npz"
    if not (f_full.exists() and f_nof.exists()):
        return False
    a, b = np.load(f_full), np.load(f_nof)
    origins = a["origins"]
    y = np.stack([d.target(t) for t in origins])
    rows = []
    for name, q in (("with fire", a["q"][..., 1]), ("without fire", b["q"][..., 1])):
        yd, pd_ = daily_means(y, q)
        err = np.abs(pd_ - yd)  # [O, 3, N]
        months = d.times[origins].to_period("M").astype(str)
        for mth in sorted(set(months)):
            sel = months == mth
            rows.append({"month": mth, "model": name, "mae": np.nanmean(err[sel])})
    df = pd.DataFrame(rows).pivot(index="month", columns="model", values="mae")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["with fire"], 0.38, color="#2a78d6", label=f"{LABELS[base]} (all inputs)")
    ax.bar(x + 0.2, df["without fire"], 0.38, color="#e87ba4", label=f"{LABELS[base]} − fire/SAI inputs")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("daily-mean MAE, days 1–3 (µg/m³)", color=INK2)
    _style(ax, f"Value of satellite fire inputs by month — {fold} test")
    ax.set_ylim(0, np.nanmax(df.to_numpy()) * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_example(cfg, fold, d: ForecastData, out, m: dict):
    """Worst-episode forecast from the fold's best model (lowest overall MAE)."""
    learned = [k for k in m if k in LABELS and "persist" not in k and k != "climatology"]
    if not learned:
        return False
    model = min(learned, key=lambda k: m[k]["overall"]["mae"])
    f = cfg.data_dir / "predictions" / f"{fold}_{model}.npz"
    if not f.exists():
        return False
    z = np.load(f)
    origins, q = z["origins"], z["q"]
    # Most polluted test origin (network mean of next 72 h), at the station with most data.
    y = np.stack([d.target(t) for t in origins])
    oi = int(np.nanargmax(np.nanmean(y, (1, 2))))
    n = int(np.argmax(np.isfinite(y[oi]).sum(0)))
    t0 = origins[oi]
    hist = slice(t0 - 72, t0 + 1)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    th = d.times[hist]
    tf = d.times[t0 + 1 : t0 + 1 + d.H]
    ax.plot(th, d.y[hist, n], color=INK2, lw=1.5, label="observed (history)")
    ax.plot(tf, y[oi, :, n], color=INK, lw=1.5, ls="--", label="observed (future)")
    ax.fill_between(tf, q[oi, :, n, 0], q[oi, :, n, -1], color=COLORS[model], alpha=0.25, lw=0, label="10–90% interval")
    ax.plot(tf, q[oi, :, n, 1], color=COLORS[model], lw=2, label=f"{LABELS[model]} median")
    ax.axvline(d.times[t0], color=INK2, lw=0.8)
    ax.set_ylabel("PM2.5 (µg/m³)", color=INK2)
    _style(ax, f"72-h forecast issued {d.times[t0]:%Y-%m-%d %H:%M} UTC — {d.station_name[n]}")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fire_season_table(d: ForecastData) -> str:
    """Daily correlation of network log-PM2.5 with fire inputs and mixing height, by burning season."""
    y = pd.Series(np.nanmean(np.log1p(d.y), 1), index=d.times)
    df = pd.DataFrame(
        {
            "y": y,
            "sai": d.sai[:, 0],
            "n24": d.fire_act[:, d.fire_names.index("fire_n_24h")],
            "blh": d.wx_an[:, :, d.wx_names.index("blh")].mean(1),
        }
    ).resample("D").mean()
    seasons = {
        "Wheat residue, Apr–May 2025": ("2025-04-01", "2025-05-31"),
        "Paddy stubble, Oct–Nov 2025": ("2025-10-01", "2025-11-30"),
        "Wheat residue, Apr–May 2026": ("2026-04-01", "2026-05-31"),
        "Paddy stubble, Oct 2026 (so far)": ("2026-10-01", "2026-10-20"),
    }
    fd = pd.DataFrame(d.fires_daily, columns=["t", "n", "frp"])
    fd.index = pd.to_datetime(fd["t"].astype("int64").astype("datetime64[h]"))
    rows = ["| Season | belt fire pixels/day | corr(PM2.5, SAI) | corr(PM2.5, fires 24 h) | corr(PM2.5, BLH) |", "|---|---|---|---|---|"]
    for name, (a, b) in seasons.items():
        x = df.loc[a:b]
        rows.append(
            f"| {name} | {fd['n'].loc[a:b].sum() / max(len(x), 1):.0f} | {x['y'].corr(x['sai']):+.2f} | "
            f"{x['y'].corr(x['n24']):+.2f} | {x['y'].corr(x['blh']):+.2f} |"
        )
    return "\n".join(rows)


def results_table(m: dict) -> str:
    head = "| Model | MAE all leads | RMSE | MAE @24h | @48h | @72h | Day-1 mean MAE | Day-1 NAQI cat. acc. | Skill vs persistence (day 1 / 2 / 3) | 80% PI coverage |\n"
    head += "|---|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for k in ORDER:
        if k not in m:
            continue
        v = m[k]
        sk = v.get("skill_vs_persistence", {})
        skill = " / ".join(f"{sk.get(d, np.nan) * 100:+.0f}%" for d in ("day1", "day2", "day3")) if sk else "—"
        cov = v.get("probabilistic", {}).get("coverage")
        cov = f"{cov * 100:.0f}%" if cov is not None and "persist" not in k and k != "climatology" else "—"
        rows.append(
            f"| {LABELS[k]} | {v['overall']['mae']:.1f} | {v['overall']['rmse']:.1f} | {v['h24']['mae']:.1f} | {v['h48']['mae']:.1f} | "
            f"{v['h72']['mae']:.1f} | {v['day1']['mae']:.1f} | {v['day1']['category_acc'] * 100:.0f}% | {skill} | {cov} |"
        )
    return head + "\n".join(rows)


def make_report(cfg: Config):
    rep = cfg.reports_dir
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fc = cfg["forecast"]
    d = ForecastData(cfg.path("processed", "dataset.npz"), fc["horizon"], fc["encoder_length"], cfg["gfs"]["availability_lag_h"])
    fig_overview(cfg, d, figs / "overview.png")
    fig_map(cfg, d, figs / "map.png")
    md = ["# Results\n", "All numbers are on held-out test periods, in µg/m³ of hourly PM2.5 unless noted. Generated by `aqf report`.\n"]
    md.append("\n## When do fires matter? (daily correlations, whole network)\n")
    md.append(fire_season_table(d) + "\n")
    for fold in cfg["split"]["folds"]:
        m = _load(cfg, fold)
        if not m:
            continue
        meta = m.pop("_meta", {})
        parts = meta.get("partitions", {})
        md.append(f"\n## Fold `{fold}`\n")
        md.append(
            f"Train {parts.get('train')} · validation {parts.get('val')} · **test {parts.get('test')}** — "
            f"{meta.get('n_stations')} stations, {meta.get('n_test_origins')} forecast origins (every 6 h), "
            f"{meta.get('test_obs_fraction', 0) * 100:.0f}% of target hours observed.\n"
        )
        md.append(results_table(m) + "\n")
        fig_horizon(cfg, fold, m, figs / f"horizon_{fold}.png")
        fig_skill(cfg, fold, m, figs / f"skill_{fold}.png")
        md.append(f"\n![MAE by lead time](figures/horizon_{fold}.png)\n\n![Skill](figures/skill_{fold}.png)\n")
        ip = rep / f"importance_{fold}.json"
        imp = json.loads(ip.read_text()) if ip.exists() else {}
        if fig_importance(cfg, fold, imp, figs / f"tft_importance_{fold}.png"):
            md.append(f"\n![TFT importance](figures/tft_importance_{fold}.png)\n")
        if fig_fire_ablation(cfg, fold, d, figs / f"fire_ablation_{fold}.png"):
            md.append(f"\n![Fire ablation](figures/fire_ablation_{fold}.png)\n")
        if fig_example(cfg, fold, d, figs / f"example_{fold}.png", m):
            md.append(f"\n![Example forecast](figures/example_{fold}.png)\n")
    (rep / "RESULTS.md").write_text("\n".join(md))
    log.info("wrote %s", rep / "RESULTS.md")
