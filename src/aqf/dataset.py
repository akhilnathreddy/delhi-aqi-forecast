"""Align stations, GFS and fires on one hourly grid and cut forecasting samples.

The build step writes ``data/processed/dataset.npz``; :class:`ForecastData`
loads it and serves, for any forecast origin ``t`` (the last hour with
observations), the three input blocks a multi-horizon model needs:

* **past** ``[L, N, F_past]`` — PM2.5 history (log-scaled, gap-filled, with a
  mask), co-pollutants, analysed weather, fire activity, SAI nowcast, calendar;
* **future** ``[H, N, F_fut]`` — GFS *forecast* weather from the most recent
  cycle that was actually available at ``t``, the SAI for each target hour
  computed from fires known at ``t``, and calendar;
* **target** ``[H, N]`` — observed PM2.5 at ``t+1 … t+H`` (NaN where missing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as fe
from .config import ROOT, Config
from .data.fires import load_fires
from .data.gfs import FIELD_NAMES
from .data.openaq import load_panel

log = logging.getLogger(__name__)

POLLUTANTS = ["pm10", "no2", "co", "o3"]
NETWORKS = ["DPCC", "CPCB", "HSPCB", "UPPCB", "IMD", "IITM", "RSPCB", "OTHER"]


# --------------------------------------------------------------------------- build
def _bilinear_weights(lat: np.ndarray, lon: np.ndarray, grid: dict) -> list[tuple[int, int, int, int, float, float]]:
    out = []
    for la, lo in zip(lat, lon):
        y, x = la - grid["lat_min"], lo - grid["lon_min"]
        i, j = int(np.floor(y)), int(np.floor(x))
        out.append((i, i + 1, j, j + 1, y - i, x - j))
    return out


def _sample_points(cube: np.ndarray, weights) -> np.ndarray:
    """``cube[..., lat, lon]`` → ``[..., P]`` bilinear samples."""
    cols = []
    for i0, i1, j0, j1, wy, wx in weights:
        v = (
            cube[..., i0, j0] * (1 - wy) * (1 - wx)
            + cube[..., i0, j1] * (1 - wy) * wx
            + cube[..., i1, j0] * wy * (1 - wx)
            + cube[..., i1, j1] * wy * wx
        )
        cols.append(v)
    return np.stack(cols, -1)


def _interp_leads(arr: np.ndarray, leads: np.ndarray, max_lead: int) -> np.ndarray:
    """3-hourly leads → hourly, NaN-aware linear interpolation along axis 0."""
    hours = np.arange(max_lead + 1)
    flat = arr.reshape(len(leads), -1)
    out = np.empty((len(hours), flat.shape[1]), np.float32)
    for k in range(flat.shape[1]):
        col = flat[:, k]
        ok = ~np.isnan(col)
        # No extrapolation past the last lead a file holds (live runs fetch short leads for past days).
        out[:, k] = np.interp(hours, leads[ok], col[ok], right=np.nan) if ok.sum() >= 2 else np.nan
    return out.reshape((len(hours),) + arr.shape[1:])


def network_of(name: str) -> str:
    for n in NETWORKS[:-1]:
        if f"- {n}" in name or f"-{n}" in name or f" {n}" in name.replace("-", " "):
            return n
    return "OTHER"


def build_dataset(cfg: Config, stations: pd.DataFrame | None = None, keep_all: bool = False) -> Path:
    """Write ``processed/dataset.npz``. ``stations``/``keep_all`` pin the station set (live runs)."""
    period, g = cfg["period"], cfg["gfs"]
    grid = g["grid"]
    times = pd.date_range(period["start"], pd.Timestamp(period["end"]) + pd.Timedelta(hours=23), freq="h")
    T = len(times)

    # ---- stations
    if stations is None:
        stations = pd.read_csv(ROOT / "configs" / "stations.csv")
    panel = load_panel(
        stations, cfg.data_dir / "raw" / "openaq", period["start"], period["end"], cfg["min_station_coverage"], keep_all
    )
    keep = list(stations["location_id"]) if keep_all else sorted(panel.index.get_level_values("location_id").unique())
    stations = stations.set_index("location_id").loc[keep].reset_index()
    N = len(stations)
    wide = panel.unstack("location_id").reindex(times)
    y = wide["pm25"][keep].to_numpy(np.float32)
    pol = np.stack([wide[p][keep].to_numpy(np.float32) for p in POLLUTANTS], -1)
    log.info("%d stations, %d hours, PM2.5 coverage %.1f%%", N, T, 100 * np.isfinite(y).mean())

    # ---- GFS: sample at stations + source-region mean, hourly, per cycle day
    gfs_dir = cfg.data_dir / "raw" / "gfs"
    w_st = _bilinear_weights(stations["lat"].to_numpy(), stations["lon"].to_numpy(), grid)
    sb = fe.SOURCE_BOX
    si = slice(int(np.ceil(sb["lat_min"] - grid["lat_min"])), int(sb["lat_max"] - grid["lat_min"]) + 1)
    sj = slice(int(np.ceil(sb["lon_min"] - grid["lon_min"])), int(sb["lon_max"] - grid["lon_min"]) + 1)
    days = pd.date_range(times[0] - pd.Timedelta(days=4), times[-1].normalize(), freq="D")
    nF = len(FIELD_NAMES)
    fc = np.full((len(days), g["max_lead"] + 1, nF, N + 1), np.nan, np.float32)  # last point = source region
    for d_i, day in enumerate(days):
        f = gfs_dir / f"{day:%Y%m%d}_{g['cycle']:02d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cube, leads = z["data"], z["leads"]  # [lead, field, lat, lon]
        pts = _sample_points(cube, w_st)  # [lead, field, N]
        src = np.nanmean(cube[:, :, si, sj], axis=(2, 3))[..., None]
        fc[d_i] = _interp_leads(np.concatenate([pts, src], -1), leads, g["max_lead"])
    have = ~np.isnan(fc[:, 6, 0, 0])
    log.info("GFS cycles available: %d / %d", have.sum(), len(days))

    # Analysis: hour h of day d comes from cycle d (leads 0-23), falling back to
    # older cycles (leads 24-47, 48-71) when a cycle is missing.
    an = np.full((T, nF, N + 1), np.nan, np.float32)
    day_index = {d: i for i, d in enumerate(days)}
    for ti, t in enumerate(times):
        d0 = day_index[t.normalize()]
        for back in range(3):
            lead = t.hour + 24 * back
            if d0 - back >= 0 and lead <= g["max_lead"] and np.isfinite(fc[d0 - back, lead, 0, 0]):
                an[ti] = fc[d0 - back, lead]
                break
    an = pd.DataFrame(an.reshape(T, -1)).interpolate(limit_direction="both").to_numpy().reshape(an.shape)
    an_f = dict(zip(FIELD_NAMES, np.moveaxis(an, 1, 0)))
    wx_an_d = fe.derive_weather({k: v[:, :N] for k, v in an_f.items()})
    wx_names = list(wx_an_d)
    wx_an = np.stack([wx_an_d[k] for k in wx_names], -1).astype(np.float32)  # [T, N, Fw]
    src_u, src_v = an_f["u850"][:, N], an_f["v850"][:, N]

    fc_f = dict(zip(FIELD_NAMES, np.moveaxis(fc, 2, 0)))
    wx_fc_d = fe.derive_weather({k: v[..., :N] for k, v in fc_f.items()})
    wx_fc = np.stack([wx_fc_d[k] for k in wx_names], -1).astype(np.float32)  # [D, 103, N, Fw]

    # ---- fires
    fires = load_fires(cfg.data_dir / "raw" / "fires")
    fires = fires[(fires["time"] >= times[0] - pd.Timedelta(days=6)) & (fires["time"] <= times[-1])]
    ext = pd.date_range(times[0] - pd.Timedelta(days=6), times[-1], freq="h")
    ext_u = pd.Series(np.interp(np.arange(len(ext)) - 144, np.arange(T), src_u), index=ext)
    ext_v = pd.Series(np.interp(np.arange(len(ext)) - 144, np.arange(T), src_v), index=ext)
    ft = fe.fire_transport(fires, ext_u, ext_v)
    sai = fe.smoke_arrival_index(ft, ext, cfg["forecast"]["horizon"])[144:]
    sai = np.log1p(sai / 10.0).astype(np.float32)
    act = fe.fire_activity(ft, ext).iloc[144:]
    log.info("fires: %d pixels (%d in source belt)", len(ft), int(ft["in_source"].sum()))

    cal = fe.calendar(times)
    out = cfg.path("processed", "dataset.npz")
    np.savez_compressed(
        out,
        times=times.values.astype("datetime64[h]").astype(np.int64),
        station_id=stations["location_id"].to_numpy(),
        station_name=stations["name"].to_numpy().astype(str),
        station_lat=stations["lat"].to_numpy(np.float32),
        station_lon=stations["lon"].to_numpy(np.float32),
        station_net=np.array([NETWORKS.index(network_of(n)) for n in stations["name"]]),
        y=y,
        pol=pol,
        wx_an=wx_an,
        wx_names=np.array(wx_names, dtype=str),
        fc_days=days.values.astype("datetime64[h]").astype(np.int64),
        fc_ok=have,
        wx_fc=wx_fc,
        fire_act=act.to_numpy(np.float32),
        fire_names=np.array(act.columns, dtype=str),
        sai=sai,
        cal=cal.to_numpy(np.float32),
        cal_names=np.array(cal.columns, dtype=str),
        fires_daily=_daily_fire_table(ft),
    )
    log.info("wrote %s", out)
    return out


def _daily_fire_table(ft: pd.DataFrame) -> np.ndarray:
    """Per-day counts in the source belt (for plots): [[day_epoch_h, n, frp_sum]]."""
    s = ft[ft["in_source"]]
    d = s.groupby(s["time"].dt.floor("D")).agg(n=("frp", "size"), frp=("frp", "sum"))
    return np.column_stack([d.index.values.astype("datetime64[h]").astype(np.int64), d["n"], d["frp"]]).astype(np.float64)


# --------------------------------------------------------------------------- samples
@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class ForecastData:
    """Model-ready view over ``dataset.npz`` with normalisation fitted on train."""

    def __init__(self, path: Path, horizon: int = 72, encoder_length: int = 168, lag_h: int = 5):
        z = np.load(path, allow_pickle=False)
        self.times = pd.to_datetime(z["times"].astype("datetime64[h]"))
        self.station_id = z["station_id"]
        self.station_name = z["station_name"]
        self.lat, self.lon, self.net = z["station_lat"], z["station_lon"], z["station_net"]
        self.y = z["y"]
        self.H, self.L, self.lag_h = horizon, encoder_length, lag_h
        self.T, self.N = self.y.shape
        self.wx_names = list(z["wx_names"])
        self.fire_names = list(z["fire_names"])
        self.cal_names = list(z["cal_names"])
        self.fc_days = pd.to_datetime(z["fc_days"].astype("datetime64[h]"))
        self.fc_ok = z["fc_ok"]
        self.wx_fc = z["wx_fc"]
        self.fires_daily = z["fires_daily"]

        # log-PM2.5 with short gaps interpolated, long gaps filled from the
        # network median at that hour (spatial imputation), plus a mask.
        logy = np.log1p(self.y)
        df = pd.DataFrame(logy).interpolate(limit=3, limit_area="inside")
        net_med = df.median(axis=1)
        df = df.apply(lambda c: c.fillna(net_med)).ffill().bfill()
        self.logy_filled = df.to_numpy(np.float32)
        self.mask = np.isfinite(self.y).astype(np.float32)
        pol = np.log1p(np.clip(z["pol"], 0, None))
        pol = np.stack([pd.DataFrame(pol[..., k]).interpolate(limit_direction="both").to_numpy() for k in range(pol.shape[-1])], -1)
        self.pol = np.nan_to_num(pol, nan=np.nanmean(pol)).astype(np.float32)
        self.wx_an = np.nan_to_num(z["wx_an"]).astype(np.float32)
        self.fire_act = z["fire_act"]
        self.sai = z["sai"]
        self.cal = z["cal"]

        # Map each origin hour to the newest GFS cycle published by then.
        day_idx = {d: i for i, d in enumerate(self.fc_days)}
        self.cycle_for = np.full(self.T, -1)
        self.lead0 = np.zeros(self.T, int)
        for ti, t in enumerate(self.times):
            avail = t - pd.Timedelta(hours=lag_h)
            d = day_idx.get(avail.normalize(), -1)
            while d >= 0 and not self.fc_ok[d]:
                d -= 1
            self.cycle_for[ti] = d
            self.lead0[ti] = int((t - self.fc_days[max(d, 0)]) / pd.Timedelta(hours=1))
        self.fit_scalers(np.arange(self.T))

    # ---------------------------------------------------------------- scaling
    def fit_scalers(self, idx: np.ndarray):
        self.mu_y = float(np.nanmean(np.log1p(self.y[idx])))
        self.sd_y = float(np.nanstd(np.log1p(self.y[idx])))
        self.mu_pol, self.sd_pol = self.pol[idx].mean((0, 1)), self.pol[idx].std((0, 1)) + 1e-6
        self.mu_wx, self.sd_wx = self.wx_an[idx].mean((0, 1)), self.wx_an[idx].std((0, 1)) + 1e-6
        self.mu_fa, self.sd_fa = self.fire_act[idx].mean(0), self.fire_act[idx].std(0) + 1e-6
        self.mu_sai, self.sd_sai = float(self.sai[idx, 0].mean()), float(self.sai[idx, 0].std() + 1e-6)

    SCALER_KEYS = ("mu_y", "sd_y", "mu_pol", "sd_pol", "mu_wx", "sd_wx", "mu_fa", "sd_fa", "mu_sai", "sd_sai")

    def scaler_state(self) -> dict:
        return {k: np.asarray(getattr(self, k)).tolist() for k in self.SCALER_KEYS}

    def load_scaler_state(self, state: dict):
        """Reuse the training normalisation (live forecasting must not refit it)."""
        for k in self.SCALER_KEYS:
            v = np.asarray(state[k], np.float32)
            setattr(self, k, float(v) if v.ndim == 0 else v)

    # ---------------------------------------------------------------- splits
    def origins(self, ranges, every_h: int = 6, gap_h: int = 72) -> np.ndarray:
        """Origins whose whole target window lies inside one of ``ranges``.

        ``ranges`` is a list of ``[start_day, end_day]`` (inclusive days). The
        first ``gap_h`` hours of a range are skipped unless it starts the data.
        """
        t = self.times
        keep = np.zeros(self.T, bool)
        if not ranges:
            return np.array([], int)
        for a, b in ranges:
            a = pd.Timestamp(a)
            b = pd.Timestamp(b) + pd.Timedelta(hours=23)
            lo = a + pd.Timedelta(hours=gap_h) if a > t[0] else a
            keep |= (t >= lo) & (t + pd.Timedelta(hours=self.H) <= b)
        keep &= t.hour % every_h == 0
        idx = np.where(keep)[0]
        idx = idx[(idx >= self.L) & (idx + self.H < self.T) & (self.cycle_for[idx] >= 0)]
        return idx[(self.lead0[idx] + self.H) < self.wx_fc.shape[1]]

    def split(self, fold: dict, gap_h: int = 72, every_h: int = 6, train_every_h: int | None = None) -> Split:
        return Split(
            self.origins(fold["train"], train_every_h or every_h, gap_h),
            self.origins(fold["val"], every_h, gap_h),
            self.origins(fold["test"], every_h, gap_h),
        )

    # ---------------------------------------------------------------- tensors
    @property
    def past_names(self) -> list[str]:
        return ["pm25", "pm25_mask"] + POLLUTANTS + self.wx_names + self.fire_names + ["sai_now"] + self.cal_names

    @property
    def future_names(self) -> list[str]:
        return [f"fc_{n}" for n in self.wx_names] + ["sai_future"] + self.cal_names

    def past(self, t: int) -> np.ndarray:
        """``[L, N, F_past]`` for history window ``(t-L, t]``."""
        s = slice(t - self.L + 1, t + 1)
        L, N = self.L, self.N
        y = (self.logy_filled[s] - self.mu_y) / self.sd_y
        pol = (self.pol[s] - self.mu_pol) / self.sd_pol
        wx = (self.wx_an[s] - self.mu_wx) / self.sd_wx
        fa = np.broadcast_to(((self.fire_act[s] - self.mu_fa) / self.sd_fa)[:, None], (L, N, len(self.fire_names)))
        sai = np.broadcast_to(((self.sai[s, 0] - self.mu_sai) / self.sd_sai)[:, None, None], (L, N, 1))
        cal = np.broadcast_to(self.cal[s][:, None], (L, N, len(self.cal_names)))
        return np.concatenate([y[..., None], self.mask[s][..., None], pol, wx, fa, sai, cal], -1).astype(np.float32)

    def future(self, t: int) -> np.ndarray:
        """``[H, N, F_fut]`` known-at-``t`` inputs for target hours ``t+1 … t+H``."""
        H, N = self.H, self.N
        d, l0 = self.cycle_for[t], self.lead0[t]
        wx = self.wx_fc[d, l0 + 1 : l0 + 1 + H]
        wx = np.nan_to_num((wx - self.mu_wx) / self.sd_wx)
        sai = np.broadcast_to(((self.sai[t, 1 : H + 1] - self.mu_sai) / self.sd_sai)[:, None, None], (H, N, 1))
        cal = np.broadcast_to(self.cal[t + 1 : t + 1 + H][:, None], (H, N, len(self.cal_names)))
        return np.concatenate([wx, sai, cal], -1).astype(np.float32)

    def target(self, t: int) -> np.ndarray:
        return self.y[t + 1 : t + 1 + self.H]

    def scale_y(self, y: np.ndarray) -> np.ndarray:
        return (np.log1p(y) - self.mu_y) / self.sd_y

    def unscale_y(self, z: np.ndarray) -> np.ndarray:
        return np.expm1(z * self.sd_y + self.mu_y)
