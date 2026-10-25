"""Parsers for the three raw data sources (no network access needed)."""

import gzip

import numpy as np
import pandas as pd

from aqf.data import fires, gfs, openaq

IDX = """1:0:d=2025011500:PRMSL:mean sea level:24 hour fcst:
2:100:d=2025011500:TMP:850 mb:24 hour fcst:
3:250:d=2025011500:UGRD:850 mb:24 hour fcst:
4:400:d=2025011500:PRATE:surface:18-24 hour ave fcst:
5:500:d=2025011500:PRATE:surface:24 hour fcst:
6:900:d=2025011500:HPBL:surface:24 hour fcst:
7:1200:d=2025011500:VIS:surface:24 hour fcst:
"""


def test_parse_idx_picks_instantaneous_messages():
    r = gfs.parse_idx(IDX)
    assert r["t850"] == (100, 249)
    assert r["u850"] == (250, 399)
    assert r["prate"] == (500, 899)  # the 6-h average message is skipped
    assert r["blh"] == (900, 1199)
    assert "t2m" not in r


def test_merge_coalesces_nearby_ranges():
    merged = gfs._merge({"a": (0, 99), "b": (100, 199), "c": (10_000_000, 10_000_100)})
    assert merged[0] == (0, 199, ["a", "b"])
    assert merged[1][2] == ["c"]


def test_footprint_and_intersection():
    hdr = b"junk geospatial_bounds POLYGON((74.0 28.0, 80.0 28.5, 80.5 33.0, 74.5 32.5, 74.0 28.0)) junk"
    poly = fires.footprint(hdr)
    assert poly.shape == (5, 2)
    box = {"lat_min": 24, "lat_max": 34, "lon_min": 68, "lon_max": 82}
    assert fires.intersects(poly, box)
    far = poly + np.array([60.0, 0.0])
    assert not fires.intersects(far, box)
    dateline = np.array([[178.9, 31.4], [-165.2, 35.1], [-148.3, 36.4], [176.4, 36.1]])
    assert not fires.intersects(dateline, box)


def test_overpass_windows_bracket_known_passes():
    # S-NPP crossed Punjab at 07:45 and 09:26 UTC and 20:28 UTC (previous evening) on 2025-11-01.
    day = pd.Timestamp("2025-11-01")
    wins = fires.candidate_windows(day, 75.0) + fires.candidate_windows(day + pd.Timedelta(days=1), 75.0)
    for t in ["2025-11-01 07:45", "2025-11-01 09:26", "2025-11-01 20:28"]:
        assert any(a <= pd.Timestamp(t) <= b for a, b in wins), t


def test_raw_to_hourly_timezone_and_min_count():
    rows = [
        {"datetime": "2025-02-19T09:30:00+05:30", "parameter": "pm25", "value": "999"},  # 03:45-04:00 UTC → hour 03
        {"datetime": "2025-02-19T09:45:00+05:30", "parameter": "pm25", "value": "100"},  # 04:00-04:15 UTC → hour 04
        {"datetime": "2025-02-19T10:00:00+05:30", "parameter": "pm25", "value": "120"},
        {"datetime": "2025-02-19T10:45:00+05:30", "parameter": "pm25", "value": "50"},  # alone in hour 05
        {"datetime": "2025-02-19T10:00:00+05:30", "parameter": "pm10", "value": "5000"},
    ]
    h = openaq.raw_to_hourly(rows)
    assert h.loc[pd.Timestamp("2025-02-19 04:00"), "pm25"] == 110
    assert "pm10" not in h or np.isnan(h.loc[pd.Timestamp("2025-02-19 04:00"), "pm10"])  # out of physical range
    assert pd.Timestamp("2025-02-19 05:00") not in h.dropna(subset=["pm25"]).index  # only 1 of 4 slots
    assert pd.Timestamp("2025-02-19 03:00") not in h.dropna(subset=["pm25"]).index


def test_flatline_masking():
    s = pd.Series([10, 11, 12, 12, 12, 12, 12, 12, 13.0])
    out = openaq.flag_flatlines(s, run=6)
    assert out.isna().sum() == 6 and out.iloc[1] == 11


def test_parse_handles_bad_gzip():
    assert openaq._parse(b"not gzip") == []
    good = gzip.compress(b'"location","lat","lon","parameter","value"\n"x","1","2","pm25","3"\n')
    assert openaq._parse(good)[0]["parameter"] == "pm25"
