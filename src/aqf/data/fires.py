"""Active-fire detections (stubble burning) from VIIRS 375 m.

Two interchangeable sources produce the same table
``[time, lat, lon, frp, confidence, satellite, daynight]``:

* :func:`download_viirs_noaa` — NOAA's Enterprise Active Fire EDR
  (``VIIRS_EFIRE_VIIRSI_EDR``) granules in the public ``noaa-nesdis-*-pds``
  buckets. This is the same I-band algorithm family that NASA FIRMS serves as
  "VIIRS 375 m", and needs no API key.
* :func:`download_firms` — the NASA FIRMS area API, for anyone with a
  ``FIRMS_MAP_KEY`` (https://firms.modaps.eosdis.nasa.gov/api/map_key/).

Global VIIRS granules are ~1 MB each and there are ~1000 per satellite per
day, so we never download blindly: every granule's footprint polygon sits in
the first ~20 kB of the netCDF header, which we read with a range request,
and only granules that intersect the fire box are fetched in full.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .http import get, s3_list, s3_url

log = logging.getLogger(__name__)

BUCKETS = {"npp": "noaa-nesdis-snpp-pds", "n20": "noaa-nesdis-n20-pds", "n21": "noaa-nesdis-n21-pds"}
PRODUCT = "VIIRS_EFIRE_VIIRSI_EDR"
HEADER_BYTES = 24_575
COLUMNS = ["time", "lat", "lon", "frp", "confidence", "satellite", "daynight"]
_POLY = re.compile(rb"POLYGON\(\(([^)]*)\)\)")
_START = re.compile(r"_s(\d{14})")
# libhdf5 is not thread-safe: decoding is serialised, downloads stay parallel.
_HDF5_LOCK = threading.Lock()


def footprint(header: bytes) -> np.ndarray | None:
    """Parse the ``geospatial_bounds`` WKT polygon → ``(n, 2)`` array of (lon, lat)."""
    m = _POLY.search(header)
    if not m:
        return None
    pts = [p.split() for p in m.group(1).decode().split(",")]
    return np.array([[float(x), float(y)] for x, y in pts])


def intersects(poly: np.ndarray, bbox: dict, margin: float = 0.5) -> bool:
    """Conservative test: does the polygon's bounding box overlap ``bbox``?

    Granules crossing the antimeridian (lon span > 180°) are never near India.
    """
    lon, lat = poly[:, 0], poly[:, 1]
    if lon.max() - lon.min() > 180:
        return False
    return not (
        lon.max() < bbox["lon_min"] - margin
        or lon.min() > bbox["lon_max"] + margin
        or lat.max() < bbox["lat_min"] - margin
        or lat.min() > bbox["lat_max"] + margin
    )


def granule_time(key: str) -> pd.Timestamp:
    s = _START.search(key).group(1)  # yyyymmddHHMMSS + tenths
    return pd.Timestamp(f"{s[:8]}T{s[8:14]}")


def candidate_windows(day: pd.Timestamp, lon_center: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """UTC windows when a 13:30-LST sun-synchronous satellite can see ``lon_center``.

    The ascending (day) node passes at ~13:30 local solar time and the
    descending (night) node at ~01:30. VIIRS's 3000 km swath reaches ±16° of
    longitude at 30°N, so the windows are padded by ±75 min.
    """
    offset = pd.Timedelta(hours=lon_center / 15.0)
    pad = pd.Timedelta(minutes=75)
    out = []
    for lst in (pd.Timedelta(hours=1, minutes=30), pd.Timedelta(hours=13, minutes=30)):
        t = day + lst - offset
        out.append((t - pad, t + pad))
    return out


def _extract(blob: bytes, bbox: dict, sat: str) -> pd.DataFrame:
    import netCDF4

    if not blob:
        return pd.DataFrame(columns=COLUMNS)
    with _HDF5_LOCK:
        return _extract_locked(netCDF4.Dataset("inmemory.nc", memory=blob), bbox, sat)


def _extract_locked(ds, bbox: dict, sat: str) -> pd.DataFrame:
    try:
        if ds.dimensions["nfire"].size == 0:
            return pd.DataFrame(columns=COLUMNS)
        lat = ds["FP_Latitude"][:].filled(np.nan)
        lon = ds["FP_Longitude"][:].filled(np.nan)
        keep = (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]) & (lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"])
        if not keep.any():
            return pd.DataFrame(columns=COLUMNS)
        t0 = pd.Timestamp(ds.getncattr("time_coverage_start")).tz_localize(None)
        return pd.DataFrame(
            {
                "time": t0,
                "lat": lat[keep],
                "lon": lon[keep],
                "frp": ds["FP_FireRadiativePower"][:].filled(np.nan)[keep],
                "confidence": ds["FP_Confidence"][:].filled(-1)[keep],
                "satellite": sat,
                "daynight": np.where(ds["FP_Day"][:].filled(0)[keep] == 1, "D", "N"),
            }
        )
    finally:
        ds.close()


def download_viirs_noaa_day(day: pd.Timestamp, sat: str, bbox: dict) -> tuple[pd.DataFrame, int, int]:
    """All fire pixels inside ``bbox`` for one satellite-day.

    Returns ``(fires, n_granules_checked, n_granules_downloaded)``.
    """
    bucket = BUCKETS[sat]
    lon_c = (bbox["lon_min"] + bbox["lon_max"]) / 2
    nxt = day + pd.Timedelta(days=1)
    windows = [w for d in (day, nxt) for w in candidate_windows(d, lon_c)]
    keys = sorted(
        key
        for key, size in s3_list(bucket, f"{PRODUCT}/{day:%Y/%m/%d}/")
        if size > 0 and key.endswith(".nc") and day <= (t := granule_time(key)) < nxt and any(a <= t <= b for a, b in windows)
    )
    with cf.ThreadPoolExecutor(16) as ex:
        polys = list(ex.map(lambda k: footprint(get(s3_url(bucket, k), (0, HEADER_BYTES))), keys))
    hits = [k for k, poly in zip(keys, polys) if poly is not None and intersects(poly, bbox)]
    frames = [_extract(get(s3_url(bucket, k)), bbox, sat) for k in hits]
    frames = [f for f in frames if len(f)]
    fires = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return fires, len(keys), len(hits)


def download_viirs_noaa(start: str, end: str, sats: list[str], bbox: dict, out_dir: Path, workers: int = 8) -> Path:
    """Download a date range, one parquet per satellite-month (resumable)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    for sat in sats:
        for month, mdays in pd.Series(days).groupby(days.to_period("M")):
            out = out_dir / f"viirs_{sat}_{month}.parquet"
            if out.exists():
                continue
            with cf.ThreadPoolExecutor(workers) as ex:
                results = list(ex.map(lambda d: download_viirs_noaa_day(d, sat, bbox), mdays))
            fires = pd.concat([r[0] for r in results], ignore_index=True)
            fires = fires.drop_duplicates(["time", "lat", "lon"])  # adjacent granules can overlap
            fires.astype({"lat": "float32", "lon": "float32", "frp": "float32"}).to_parquet(out)
            log.info(
                "%s %s: %d fire pixels (%d granules checked, %d downloaded)",
                sat, month, len(fires), sum(r[1] for r in results), sum(r[2] for r in results),
            )
    return out_dir


def download_firms(start: str, end: str, bbox: dict, out_dir: Path, source: str = "VIIRS_SNPP_SP") -> Path:
    """NASA FIRMS area API (needs ``FIRMS_MAP_KEY``); 10-day chunks, same schema."""
    key = os.environ["FIRMS_MAP_KEY"]
    area = f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"
    frames = []
    for chunk in pd.date_range(start, end, freq="10D"):
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/10/{chunk:%Y-%m-%d}"
        raw = get(url)
        if not raw:
            continue
        d = pd.read_csv(io.BytesIO(raw))
        hhmm = d["acq_time"].astype(int).astype(str).str.zfill(4)
        frames.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(d["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M"),
                    "lat": d["latitude"],
                    "lon": d["longitude"],
                    "frp": d["frp"],
                    "confidence": d["confidence"].map({"l": 7, "n": 8, "h": 9}).fillna(-1),
                    "satellite": source,
                    "daynight": d["daynight"],
                }
            )
        )
    out = Path(out_dir) / f"firms_{source}_{start}_{end}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out)
    return out


def load_fires(fire_dir: Path) -> pd.DataFrame:
    files = sorted(Path(fire_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("time")
