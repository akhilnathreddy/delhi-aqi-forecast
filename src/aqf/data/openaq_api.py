"""Recent hourly PM2.5 from the OpenAQ v3 API (optional, needs a free API key).

The AWS archive the training data comes from is refreshed only every 2-4
days, so on its own it cannot drive a forecast issued "now". With an
``OPENAQ_API_KEY`` the live pipeline tops up the last few days of PM2.5 from
``/v3/sensors/{id}/hours``. The sensor ids come from the archive files
(``sensors_id`` column) and are stored in the model bundle.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import requests

API = "https://api.openaq.org/v3"
_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL_S = 1.05  # free tier: 60 requests per minute


def _throttled_get(url: str, params: dict, api_key: str, timeout: float = 30) -> dict:
    with _lock:
        wait = _last_call[0] + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
    r = requests.get(url, params=params, headers={"X-API-Key": api_key}, timeout=timeout)
    if r.status_code == 429:  # rate limited: back off once
        time.sleep(float(r.headers.get("Retry-After", 5)))
        r = requests.get(url, params=params, headers={"X-API-Key": api_key}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_hours(payload: dict) -> pd.Series:
    """``/sensors/{id}/hours`` JSON → hourly values indexed by period start (UTC, naive)."""
    vals, idx = [], []
    for row in payload.get("results", []):
        start = (row.get("period") or {}).get("datetimeFrom", {}).get("utc")
        if start is None or row.get("value") is None:
            continue
        idx.append(pd.Timestamp(start).tz_convert(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start))
        vals.append(float(row["value"]))
    s = pd.Series(vals, index=pd.DatetimeIndex(idx, name="time"), dtype=float).sort_index()
    return s[~s.index.duplicated()]


def sensor_hours(sensor_id: int, start: pd.Timestamp, end: pd.Timestamp, api_key: str) -> pd.Series:
    payload = _throttled_get(
        f"{API}/sensors/{sensor_id}/hours",
        {"datetime_from": f"{start:%Y-%m-%dT%H:%M:%SZ}", "datetime_to": f"{end:%Y-%m-%dT%H:%M:%SZ}", "limit": 1000},
        api_key,
    )
    return parse_hours(payload)
