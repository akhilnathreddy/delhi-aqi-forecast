"""Issue a forecast now: fetch the latest data window, rebuild inputs, run the bundle.

Everything is downloaded into a scratch directory with the same downloaders
used for training, then :func:`aqf.dataset.build_dataset` assembles a small
dataset (the last ~12 days plus 4 days of future calendar/GFS) pinned to the
bundle's station order, and the bundled models forecast from the latest
hour with enough observations.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data import gfs, openaq
from .data.fires import download_viirs_noaa
from .dataset import ForecastData, build_dataset
from .serve import Bundle

log = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


@dataclass
class LiveForecast:
    issued: pd.Timestamp  # forecast origin (last hour used), UTC
    now: pd.Timestamp
    stations: pd.DataFrame  # location_id, name, lat, lon, last_obs
    hist_times: pd.DatetimeIndex
    hist_y: np.ndarray  # [L_hist, N] observed PM2.5
    fc_times: pd.DatetimeIndex
    forecasts: dict[str, np.ndarray]  # model → [H, N, Q] µg/m³
    quantiles: list[float]
    sai: np.ndarray  # smoke-arrival index for each forecast hour (log scale)
    weather: pd.DataFrame  # network-mean GFS forecast for the forecast hours
    fires_daily: pd.Series  # source-belt fire pixels per day
    obs_until: pd.Timestamp | None = None  # last hour with real observations (≤ issued)
    gap_times: pd.DatetimeIndex | None = None  # hours filled with the model's own forecast
    gap_y: np.ndarray | None = None  # [gap, N] model-filled PM2.5
    notes: list[str] = field(default_factory=list)


def _download_stations(bundle: Bundle, raw: Path, start, end, api_key: str | None, progress: Progress):
    out = raw / "openaq"
    st = bundle.stations
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(openaq.download_station, int(i), str(start.date()), str(end.date()), out, 8): i for i in st["location_id"]}
        for k, f in enumerate(cf.as_completed(futs)):
            try:
                f.result()
            except Exception as e:  # one station must not stop the forecast
                log.warning("station %s failed: %s", futs[f], e)
            progress(f"Station data {k + 1}/{len(st)}", 0.05 + 0.25 * (k + 1) / len(st))
    if not api_key:
        return
    from .data.openaq_api import sensor_hours

    sensors = bundle.meta.get("sensors", {})
    for k, loc in enumerate(st["location_id"]):
        sid = sensors.get(str(int(loc)), {}).get("pm25")
        if not sid:
            continue
        f = out / f"{int(loc)}.parquet"
        d = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["pm25"])
        last = d["pm25"].dropna().index.max() if "pm25" in d and d["pm25"].notna().any() else start
        try:
            recent = sensor_hours(int(sid), last, end, api_key)
        except Exception as e:
            log.warning("OpenAQ API failed for %s: %s", loc, e)
            continue
        recent = recent[(recent > 0) & (recent < 999)]
        d = d.reindex(d.index.union(recent.index))
        d["pm25"] = d["pm25"].combine_first(recent) if "pm25" in d else recent
        d.index.name = "time"
        d.to_parquet(f)
        progress(f"Recent PM2.5 via OpenAQ API {k + 1}/{len(st)}", 0.30 + 0.10 * (k + 1) / len(st))


def run_live(
    bundle: Bundle,
    workdir: Path,
    now: pd.Timestamp | None = None,
    api_key: str | None = None,
    progress: Progress = lambda msg, frac: None,
    cfg: Config | None = None,
) -> LiveForecast:
    cfg = cfg or load_config()
    now = (now or pd.Timestamp.now(tz="UTC").tz_convert(None)).floor("h")
    # 7 days of encoder history before an origin that may itself be up to ~4 days
    # old (archive lag), plus a day of slack.
    start = (now - pd.Timedelta(days=12)).normalize()
    end = (now + pd.Timedelta(days=4)).normalize()
    raw = Path(workdir) / "raw"
    live_cfg = Config(copy.deepcopy(cfg.raw), root=cfg.root)
    live_cfg.raw["data_dir"] = str(Path(workdir).resolve())
    live_cfg.raw["period"] = {"start": str(start.date()), "end": str(end.date())}
    notes = []

    progress("Downloading station data (CPCB via OpenAQ archive)", 0.02)
    _download_stations(bundle, raw, start, now, api_key, progress)

    progress("Downloading GFS forecasts", 0.40)
    g = cfg["gfs"]
    days = pd.date_range(start - pd.Timedelta(days=4), now.normalize(), freq="D")
    recent = now.normalize() - pd.Timedelta(days=1)
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda d: gfs.download_day(d, raw / "gfs", g["cycle"], g["max_lead"] if d >= recent else 24, g["grid"]), days))

    progress("Downloading VIIRS fire detections", 0.60)
    try:
        download_viirs_noaa(str((start - pd.Timedelta(days=7)).date()), str(now.date()), cfg["fire_satellites"], cfg["fire_bbox"], raw / "fires")
    except Exception as e:  # the forecast still runs; the smoke inputs just read as "no fires"
        log.warning("fire download failed: %s", e)
        notes.append("Satellite fire data could not be downloaded, so this forecast assumes no stubble smoke is on its way.")

    progress("Building model inputs", 0.85)
    path = build_dataset(live_cfg, stations=bundle.stations[["location_id", "name", "lat", "lon"]], keep_all=True)
    fc = cfg["forecast"]
    d = ForecastData(path, fc["horizon"], fc["encoder_length"], g["availability_lag_h"])

    # Observations end at the newest hour (≤ now) with PM2.5 from at least 30% of stations.
    idx = np.arange(d.T)
    usable = (idx >= d.L) & (idx + d.H < d.T) & (d.cycle_for >= 0) & (d.lead0 + d.H < d.wx_fc.shape[1])
    frac = np.isfinite(d.y).mean(1)
    obs_ok = usable & (d.times <= now) & (frac >= 0.3)
    if not obs_ok.any():
        raise RuntimeError("Not enough recent station or weather data to issue a forecast. The data sources may be down; try again later.")
    t_obs = int(np.where(obs_ok)[0][-1])
    now_ok = usable & (d.times <= now)
    t = int(np.where(now_ok)[0][-1]) if now_ok.any() else t_obs

    progress("Running models", 0.92)
    gap_times, gap_y = None, None
    if t - t_obs > 3:
        # Stale observations: roll the model forward across the gap and use its median
        # as provisional history (flagged unobserved via the mask), then issue from now.
        gap = t - t_obs
        filled = np.empty((gap, d.N), np.float32)
        d.logy_filled = d.logy_filled.copy()  # pandas hands back a read-only view
        cur = t_obs
        while cur < t:
            preds = bundle.predict(d, np.array([cur]))
            med = preds.get("ensemble_anchored", next(iter(preds.values())))[0][..., 1]
            span = min(d.H, t - cur)
            d.logy_filled[cur + 1 : cur + 1 + span] = np.log1p(np.clip(med[:span], 0, None))
            filled[cur - t_obs : cur - t_obs + span] = med[:span]
            cur += span
        gap_times, gap_y = d.times[t_obs + 1 : t + 1], filled
        notes.append(
            f"Station observations end {gap} h before the forecast time (the public archive updates every 2–4 days). "
            "The gap was filled with the model's own forecast, so expect larger errors than in the backtests. "
            "Set the OPENAQ_API_KEY secret for near-real-time PM2.5."
        )
    forecasts = bundle.predict(d, np.array([t]))
    forecasts = {k: v[0] for k, v in forecasts.items()}

    last_obs = [d.times[np.where(np.isfinite(d.y[: t + 1, n]))[0][-1]] if np.isfinite(d.y[: t + 1, n]).any() else pd.NaT for n in range(d.N)]
    st = bundle.stations.copy()
    st["last_obs"] = last_obs
    hist = slice(max(0, t_obs - 7 * 24 + 1), t_obs + 1)
    k_l, k_w = d.cycle_for[t], d.lead0[t]
    wx = d.wx_fc[k_l, k_w + 1 : k_w + 1 + d.H]  # [H, N, Fw] raw units
    with np.errstate(all="ignore"):
        weather = pd.DataFrame(np.nanmean(wx, 1), columns=d.wx_names, index=d.times[t + 1 : t + 1 + d.H])
    fd = pd.Series(d.fires_daily[:, 1], index=pd.to_datetime(d.fires_daily[:, 0].astype("int64").astype("datetime64[h]")))
    progress("Done", 1.0)
    return LiveForecast(
        issued=d.times[t],
        now=now,
        stations=st,
        hist_times=d.times[hist],
        hist_y=d.y[hist],
        fc_times=d.times[t + 1 : t + 1 + d.H],
        forecasts=forecasts,
        quantiles=bundle.quantiles,
        sai=d.sai[t, 1 : d.H + 1],
        weather=weather,
        fires_daily=fd,
        obs_until=d.times[t_obs],
        gap_times=gap_times,
        gap_y=gap_y,
        notes=notes,
    )
