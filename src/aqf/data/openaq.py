"""CPCB / state-board station data via the OpenAQ AWS archive.

CPCB's CCR portal has no bulk API, but OpenAQ ingests every CPCB, DPCC,
HSPCB, UPPCB, IMD and IITM continuous station and republishes the raw
15-minute readings as one gzipped CSV per station-day in the public
``openaq-data-archive`` bucket::

    records/csv.gz/locationid={id}/year={yyyy}/month={mm}/location-{id}-{yyyymmdd}.csv.gz

This module discovers the stations inside the NCR box, downloads their
station-days in parallel and reduces them to a QC'd hourly table.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import gzip
import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .http import get, s3_list, s3_url

log = logging.getLogger(__name__)

BUCKET = "openaq-data-archive"
PREFIX = "records/csv.gz/"
PARAMETERS = ["pm25", "pm10", "no2", "no", "co", "o3", "so2", "temperature", "relativehumidity"]
# CPCB analysers saturate at 999 µg/m³ for PM2.5; anything above is an artefact.
VALID_RANGE = {"pm25": (1.0, 999.0), "pm10": (1.0, 1500.0)}


def list_location_ids() -> list[int]:
    ids = {int(m.group(1)) for p, _ in s3_list(BUCKET, PREFIX, "/") if (m := re.search(r"locationid=(\d+)/", p))}
    return sorted(ids)


def _sample_location(loc_id: int, years: tuple[int, ...]) -> dict | None:
    for year in years:
        keys = s3_list(BUCKET, f"{PREFIX}locationid={loc_id}/year={year}/")
        key = next((k for k, size in keys if size > 0), None)
        if key is None:
            continue
        rows = _parse(get(s3_url(BUCKET, key)))
        if not rows:
            continue
        r = rows[0]
        return {
            "location_id": loc_id,
            "name": re.sub(r"-\d+$", "", r["location"]).strip(),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "parameters": "|".join(sorted({x["parameter"] for x in rows})),
            "first_year_seen": year,
        }
    return None


def discover_stations(bbox: dict, years: tuple[int, ...] = (2025, 2026), workers: int = 64) -> pd.DataFrame:
    """Scan every archive location and keep those inside ``bbox`` reporting PM2.5.

    This touches ~56k locations and takes 15-25 minutes; the result is
    committed as ``configs/stations.csv`` so normal runs skip it.
    """
    ids = list_location_ids()
    log.info("scanning %d OpenAQ locations", len(ids))
    out = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for meta in ex.map(lambda i: _sample_location(i, years), ids):
            if meta is None:
                continue
            inside = bbox["lat_min"] <= meta["lat"] <= bbox["lat_max"] and bbox["lon_min"] <= meta["lon"] <= bbox["lon_max"]
            if inside and "pm25" in meta["parameters"].split("|"):
                out.append(meta)
    return pd.DataFrame(out).sort_values("location_id").reset_index(drop=True)


def _parse(blob: bytes) -> list[dict]:
    if not blob:
        return []
    try:
        text = gzip.decompress(blob).decode("utf-8", "replace")
    except OSError:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def _station_keys(loc_id: int, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    keys = []
    for year in range(start.year, end.year + 1):
        for key, size in s3_list(BUCKET, f"{PREFIX}locationid={loc_id}/year={year}/"):
            m = re.search(r"-(\d{8})\.csv\.gz$", key)
            if size > 0 and m and start <= pd.Timestamp(m.group(1)) <= end:
                keys.append(key)
    return keys


def raw_to_hourly(rows: list[dict]) -> pd.DataFrame:
    """15-minute long rows → hourly wide frame (UTC, period-beginning labels).

    An hour is kept only if at least two of its four 15-minute slots are present.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["datetime", "parameter", "value"])
    df = df[df["parameter"].isin(PARAMETERS)]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["time"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    # A reading stamped 04:00 covers 03:45-04:00, so it belongs to hour 03:00.
    df["hour"] = (df["time"] - pd.Timedelta(seconds=1)).dt.floor("h").dt.tz_localize(None)
    for p, (lo, hi) in VALID_RANGE.items():
        bad = (df["parameter"] == p) & ~df["value"].between(lo, hi)
        df.loc[bad, "value"] = np.nan
    g = df.dropna(subset=["value"]).groupby(["hour", "parameter"])["value"]
    hourly = g.mean().where(g.count() >= 2).unstack("parameter")
    hourly.index.name = "time"
    return hourly.sort_index()


def flag_flatlines(s: pd.Series, run: int = 6) -> pd.Series:
    """Mask runs of >= ``run`` identical consecutive hourly values (stuck analyser)."""
    same = s.diff().eq(0)
    grp = (~same).cumsum()
    length = same.groupby(grp).transform("sum") + 1
    return s.mask(length >= run)


def download_station(loc_id: int, start: str, end: str, out_dir: Path, workers: int = 32) -> Path:
    out = Path(out_dir) / f"{loc_id}.parquet"
    if out.exists():
        return out
    keys = _station_keys(loc_id, pd.Timestamp(start), pd.Timestamp(end))
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for part in ex.map(lambda k: _parse(get(s3_url(BUCKET, k))), keys):
            rows.extend(part)
    hourly = raw_to_hourly(rows)
    if "pm25" in hourly:
        hourly["pm25"] = flag_flatlines(hourly["pm25"])
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out)
    log.info("station %s: %d station-days → %d hourly rows", loc_id, len(keys), len(hourly))
    return out


def load_panel(stations: pd.DataFrame, raw_dir: Path, start: str, end: str, min_coverage: float) -> pd.DataFrame:
    """Stack station files into a (time × station) PM2.5 panel plus co-pollutants.

    Returns a long frame indexed by (time, location_id) on a complete hourly
    grid, and drops stations whose PM2.5 coverage is below ``min_coverage``.
    """
    grid = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    frames = []
    for loc in stations["location_id"]:
        f = Path(raw_dir) / f"{loc}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f).reindex(grid)
        if "pm25" not in d or d["pm25"].notna().mean() < min_coverage:
            log.info("dropping station %s (pm25 coverage %.2f)", loc, d.get("pm25", pd.Series(dtype=float)).notna().mean())
            continue
        d = d.reindex(columns=PARAMETERS)
        d["location_id"] = loc
        frames.append(d.rename_axis("time").reset_index())
    return pd.concat(frames).set_index(["time", "location_id"]).sort_index()
