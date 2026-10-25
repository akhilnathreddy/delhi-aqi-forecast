"""NOAA GFS 1.0° forecasts from the ``noaa-gfs-bdp-pds`` AWS Open Data bucket.

Each forecast file ships with a ``.idx`` sidecar listing the byte offset of
every GRIB2 message, so we fetch only the ~10 fields we need with HTTP range
requests (about 0.7 MB per lead instead of 45 MB) and crop them to a small
regional grid covering NCR and the upwind fire region.

Using *forecast* weather (not observed weather) for the future part of each
sample is what makes the setup honest: at forecast time nobody knows the
real wind 48 h ahead, only what the model said.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .http import get

log = logging.getLogger(__name__)

BUCKET_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
_ECCODES_LOCK = threading.Lock()  # GRIB decoding is serialised; downloads stay parallel


@dataclass(frozen=True)
class Field:
    name: str
    var: str
    level: str


FIELDS = (
    Field("t2m", "TMP", "2 m above ground"),
    Field("rh2m", "RH", "2 m above ground"),
    Field("u10", "UGRD", "10 m above ground"),
    Field("v10", "VGRD", "10 m above ground"),
    Field("blh", "HPBL", "surface"),
    Field("prate", "PRATE", "surface"),
    Field("tcc", "TCDC", "entire atmosphere"),
    Field("t850", "TMP", "850 mb"),
    Field("u850", "UGRD", "850 mb"),
    Field("v850", "VGRD", "850 mb"),
)
FIELD_NAMES = [f.name for f in FIELDS]


def file_url(day: pd.Timestamp, cycle: int, lead: int) -> str:
    d = day.strftime("%Y%m%d")
    return f"{BUCKET_URL}/gfs.{d}/{cycle:02d}/atmos/gfs.t{cycle:02d}z.pgrb2.1p00.f{lead:03d}"


def parse_idx(text: str) -> dict[str, tuple[int, int]]:
    """Map field name → inclusive byte range of its *instantaneous* message.

    Averaged/accumulated variants (``0-6 hour ave fcst``) are skipped; at
    f000 GFS has no instantaneous PRATE/TCDC, and those stay missing.
    """
    lines = [ln.split(":") for ln in text.strip().splitlines() if ln]
    offsets = [int(p[1]) for p in lines]
    ranges: dict[str, tuple[int, int]] = {}
    for i, parts in enumerate(lines):
        var, level, step = parts[3], parts[4], parts[5]
        if any(w in step for w in ("ave", "acc", "max", "min")):
            continue
        for f in FIELDS:
            if f.var == var and f.level == level and f.name not in ranges:
                end = offsets[i + 1] - 1 if i + 1 < len(offsets) else -1
                ranges[f.name] = (offsets[i], end)
    return ranges


def _merge(ranges: dict[str, tuple[int, int]], max_gap: int = 200_000) -> list[tuple[int, int, list[str]]]:
    """Coalesce nearby byte ranges into fewer HTTP requests."""
    items = sorted(ranges.items(), key=lambda kv: kv[1][0])
    merged: list[tuple[int, int, list[str]]] = []
    for name, (s, e) in items:
        if merged and e != -1 and merged[-1][1] != -1 and s - merged[-1][1] <= max_gap:
            ps, _, names = merged[-1]
            merged[-1] = (ps, e, names + [name])
        else:
            merged.append((s, e, [name]))
    return merged


def decode_messages(blob: bytes, grid: dict) -> dict[int, np.ndarray]:
    """Decode concatenated GRIB2 messages, cropped to ``grid`` (1° lat/lon box).

    Returns ``{byte offset of message in blob: (n_lat, n_lon) array}`` with
    latitude increasing south → north.
    """
    import eccodes  # imported lazily: only the downloader needs the C library

    out: dict[int, np.ndarray] = {}
    pos = 0
    while (start := blob.find(b"GRIB", pos)) >= 0:
        length = int.from_bytes(blob[start + 8 : start + 16], "big")
        if length == 0 or start + length > len(blob):
            break
        msg = eccodes.codes_new_from_message(blob[start : start + length])
        try:
            ni, nj = eccodes.codes_get(msg, "Ni"), eccodes.codes_get(msg, "Nj")
            vals = eccodes.codes_get_values(msg).reshape(nj, ni)
            lat0 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
        finally:
            eccodes.codes_release(msg)
        lats = lat0 - np.arange(nj)  # GFS 1.0° rows run 90 → -90
        lons = np.arange(ni)
        li = (lats >= grid["lat_min"]) & (lats <= grid["lat_max"])
        lj = (lons >= grid["lon_min"]) & (lons <= grid["lon_max"])
        out[start] = vals[np.ix_(li, lj)][::-1].astype(np.float32)
        pos = start + length
    return out


def fetch_lead(day: pd.Timestamp, cycle: int, lead: int, grid: dict) -> np.ndarray | None:
    """Return a ``(n_fields, n_lat, n_lon)`` array for one forecast lead, NaN where absent."""
    url = file_url(day, cycle, lead)
    idx = get(url + ".idx").decode()
    if not idx:
        return None
    ranges = parse_idx(idx)
    nlat = grid["lat_max"] - grid["lat_min"] + 1
    nlon = grid["lon_max"] - grid["lon_min"] + 1
    arr = np.full((len(FIELDS), nlat, nlon), np.nan, np.float32)
    for s, e, names in _merge(ranges):
        blob = get(url, (s, e if e != -1 else s + 5_000_000))
        with _ECCODES_LOCK:
            decoded = decode_messages(blob, grid)
        for name in names:
            field = decoded.get(ranges[name][0] - s)
            if field is not None:
                arr[FIELD_NAMES.index(name)] = field
    return arr


def download_day(day: pd.Timestamp, out_dir: Path, cycle: int, max_lead: int, grid: dict, workers: int = 16) -> Path | None:
    out = Path(out_dir) / f"{day:%Y%m%d}_{cycle:02d}.npz"
    if out.exists():
        return out
    leads = list(range(0, max_lead + 1, 3))

    def safe(ld):
        try:
            return fetch_lead(day, cycle, ld, grid)
        except Exception as e:  # one bad lead must not lose the whole day
            log.warning("GFS %s f%03d failed: %s", day.date(), ld, e)
            return None

    with cf.ThreadPoolExecutor(workers) as ex:
        arrays = list(ex.map(safe, leads))
    if all(a is None for a in arrays):
        log.warning("GFS %s %02dZ missing", day.date(), cycle)
        return None
    shape = next(a.shape for a in arrays if a is not None)
    stack = np.stack([a if a is not None else np.full(shape, np.nan, np.float32) for a in arrays])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=stack, leads=np.array(leads), fields=np.array(FIELD_NAMES))
    return out
