"""Delhi-NCR PM2.5 forecaster — Streamlit app.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud → this repo → main file ``streamlit_app.py``.
Optional secret ``OPENAQ_API_KEY`` gives near-real-time PM2.5 for live forecasts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

IST = pd.Timedelta(hours=5, minutes=30)
NAQI = [  # (upper bound µg/m³, label, colour) — CPCB National AQI bands for 24-h PM2.5
    (30, "Good", "#2e7d32"),
    (60, "Satisfactory", "#8bc34a"),
    (90, "Moderate", "#f2c12e"),
    (120, "Poor", "#f08a24"),
    (250, "Very Poor", "#d83a2e"),
    (np.inf, "Severe", "#7b1c1c"),
]
MODEL_LABELS = {
    "ensemble_anchored": "Ensemble (level-anchored)",
    "lightgbm_resid": "LightGBM",
    "tft_anchored": "Temporal Fusion Transformer",
    "stgnn_anchored": "Wind-aware graph network",
    "ensemble": "Ensemble (level)",
    "persistence": "Persistence",
    "diurnal_persistence": "Same hour, last day",
    "climatology": "Climatology",
    "lightgbm": "LightGBM (level)",
    "lightgbm_nofire": "LightGBM (level), no fire inputs",
    "lightgbm_nofcwx": "LightGBM (level), no weather forecast",
    "lightgbm_resid_nofire": "LightGBM, no fire inputs",
    "tft": "Temporal Fusion Transformer (level)",
    "stgnn": "Wind-aware graph network (level)",
}

MODEL_COLORS = {
    "persistence": "#9b9a96",
    "lightgbm_resid": "#2a78d6",
    "tft_anchored": "#eb6834",
    "stgnn_anchored": "#1baf7a",
    "ensemble": "#6250d6",
    "ensemble_anchored": "#e34948",
}

st.set_page_config(page_title="Delhi-NCR PM2.5 Forecast", page_icon="🌫️", layout="wide")


# --------------------------------------------------------------------------- helpers
def naqi(v: float) -> tuple[str, str]:
    if v is None or not np.isfinite(v):
        return "No data", "#9e9e9e"
    for hi, label, color in NAQI:
        if v <= hi:
            return label, color
    return NAQI[-1][1], NAQI[-1][2]


def hex_rgb(h: str) -> list[int]:
    return [int(h[i : i + 2], 16) for i in (1, 3, 5)]


def to_ist(t) -> pd.DatetimeIndex | pd.Timestamp:
    return t + IST


def fan_chart(hist: pd.DataFrame | None, fc: pd.DataFrame, title: str, height: int = 320) -> alt.Chart:
    """Observed history (+ optional observed future) and forecast median with a 10–90% band."""
    x = alt.X("time:T", title="Time (IST)", axis=alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=8))
    y_scale = alt.Scale(zero=True)
    layers = [
        alt.Chart(fc).mark_area(opacity=0.22, color="#1f5c8f").encode(
            x=x, y=alt.Y("q10:Q", title="PM2.5 (µg/m³)", scale=y_scale), y2="q90:Q",
            tooltip=[alt.Tooltip("time:T", format="%d %b %H:%M"), alt.Tooltip("q10:Q", format=".0f", title="10%"),
                     alt.Tooltip("q50:Q", format=".0f", title="median"), alt.Tooltip("q90:Q", format=".0f", title="90%")],
        ),
        alt.Chart(fc).mark_line(color="#1f5c8f", strokeWidth=2.2).encode(x=x, y="q50:Q"),
    ]
    if hist is not None and len(hist):
        styles = {"observed": [1, 0], "observed (after issue)": [5, 3], "model-filled gap": [1, 3]}
        kinds = [k for k in styles if k in set(hist["kind"])]
        layers.insert(0, alt.Chart(hist).mark_line(color="#4b5560", strokeWidth=1.5).encode(
            x=x, y="pm25:Q",
            strokeDash=alt.StrokeDash("kind:N", legend=alt.Legend(title=None, orient="top", direction="horizontal"),
                                      scale=alt.Scale(domain=kinds, range=[styles[k] for k in kinds])),
            tooltip=[alt.Tooltip("time:T", format="%d %b %H:%M"), alt.Tooltip("pm25:Q", format=".0f", title="observed")],
        ))
    return alt.layer(*layers).properties(title=title, height=height)


# --------------------------------------------------------------------------- data loaders
@st.cache_resource(show_spinner=False)
def load_bundle():
    from aqf.serve import Bundle

    return Bundle()


@st.cache_data(show_spinner=False)
def load_backtest():
    z = np.load(ROOT / "app" / "backtest.npz")
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def load_metrics(fold: str) -> dict:
    p = ROOT / "reports" / f"metrics_{fold}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:
        return os.environ.get(name)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def live_forecast(hour_bucket: str, use_api: bool):
    """Cached for 3 h per bucket so repeated visits don't re-download anything."""
    from aqf.live import run_live

    bar = st.progress(0.0, text="Starting…")
    with tempfile.TemporaryDirectory() as tmp:
        res = run_live(
            load_bundle(), Path(tmp), api_key=_secret("OPENAQ_API_KEY") if use_api else None,
            progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg),
        )
    bar.empty()
    return res


# --------------------------------------------------------------------------- pages
def page_live():
    st.subheader("Next 72 hours")
    bundle_ok = (ROOT / "models" / "final" / "meta.json").exists()
    if not bundle_ok:
        st.error("Model bundle not found in models/final. Run `python -m aqf.cli train-final` first.")
        return
    has_key = bool(_secret("OPENAQ_API_KEY"))
    st.caption(
        "Fetches the latest CPCB station data, the newest GFS forecast and VIIRS fire detections, then runs the "
        "level-anchored ensemble (LightGBM + Temporal Fusion Transformer + wind-aware graph network). "
        "A fresh run takes 1–3 minutes; results are cached for 3 hours."
        + ("" if has_key else " Without an OpenAQ API key the station archive lags 2–4 days; the app bridges that gap with the model's own forecast and marks it on the charts.")
    )
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    bucket = str(now.floor("3h"))
    if "live_requested" not in st.session_state:
        st.session_state.live_requested = False
    if st.button("Run forecast", type="primary") or st.session_state.live_requested:
        st.session_state.live_requested = True
        try:
            res = live_forecast(bucket, has_key)
        except Exception as e:
            st.error(f"The forecast could not be issued: {e}")
            return
        render_live(res)
    else:
        st.info("Press **Run forecast** to fetch today's data and forecast every station.")


def history_frame(res, n: int | None) -> pd.DataFrame:
    """Observed history plus, when observations were stale, the model-filled gap (station n or network mean)."""
    pick = (lambda a: np.nanmean(a, 1)) if n is None else (lambda a: a[:, n])
    with np.errstate(all="ignore"):
        parts = [pd.DataFrame({"time": to_ist(res.hist_times), "pm25": pick(res.hist_y), "kind": "observed"})]
        if res.gap_times is not None:
            parts.append(pd.DataFrame({"time": to_ist(res.gap_times), "pm25": pick(res.gap_y), "kind": "model-filled gap"}))
    return pd.concat(parts, ignore_index=True)


def chip(label: str, color: str) -> str:
    dark = label in ("Satisfactory", "Moderate", "Poor")
    return f"<span style='background:{color};color:{'#111' if dark else '#fff'};padding:2px 8px;font-size:0.85em'>{label}</span>"


def render_live(res):
    model = st.radio("Model", [m for m in MODEL_LABELS if m in res.forecasts], format_func=MODEL_LABELS.get, horizontal=True)
    q = res.forecasts[model]  # [H, N, Q]
    med = q[..., 1]
    st_df = res.stations.copy()
    for k, sl in enumerate((slice(0, 24), slice(24, 48), slice(48, 72))):
        st_df[f"day{k + 1}"] = med[sl].mean(0)

    for n in res.notes:
        st.warning(n)
    issued = to_ist(res.issued)
    c = st.columns(4)
    c[0].metric("Forecast issued", f"{issued:%d %b %H:%M} IST")
    if res.gap_times is not None:
        c[0].caption(f"Observations up to {to_ist(res.obs_until):%d %b %H:%M} IST")
    for k in range(3):
        v = float(np.nanmean(st_df[f"day{k + 1}"]))
        day0 = (issued + pd.Timedelta(hours=24 * k + 1)).strftime("%d %b %H:%M")
        c[k + 1].metric(f"Network mean, day {k + 1}", f"{v:.0f} µg/m³")
        c[k + 1].markdown(f"{chip(*naqi(v))} &nbsp;<small>24 h from {day0}</small>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        day = st.select_slider("Map shows", ["day1", "day2", "day3"], format_func=lambda s: f"Day {s[-1]} mean")
        st_df["category"] = [naqi(v)[0] for v in st_df[day]]
        st_df["color"] = [hex_rgb(naqi(v)[1]) for v in st_df[day]]
        st_df["value"] = st_df[day].round(0)
        st.pydeck_chart(
            pdk.Deck(
                layers=[pdk.Layer("ScatterplotLayer", st_df, get_position="[lon, lat]", get_fill_color="color",
                                  get_radius=1400, pickable=True, opacity=0.85, stroked=True, get_line_color=[255, 255, 255])],
                initial_view_state=pdk.ViewState(latitude=28.6, longitude=77.2, zoom=8.6),
                tooltip={"text": "{name}\n{value} µg/m³ · {category}"},
                map_style=None,
            ),
            height=430,
        )
        legend = " ".join(chip(label, color) for _, label, color in NAQI)
        st.markdown(legend, unsafe_allow_html=True)
    with right:
        net_hist = history_frame(res, None)
        net_fc = pd.DataFrame({"time": to_ist(res.fc_times), "q10": np.nanmean(q[..., 0], 1), "q50": med.mean(1), "q90": np.nanmean(q[..., 2], 1)})
        st.altair_chart(fan_chart(net_hist, net_fc, "Network mean PM2.5"), width="stretch", height=360)
        w = res.weather
        drivers = pd.DataFrame({"time": to_ist(w.index), "blh": w["blh"].values, "ws10": w["ws10"].values, "sai": res.sai})
        for col, title, fmt in (("blh", "Mixing-layer height, m (low = pollution trapped)", ",.0f"),
                                ("ws10", "Wind at 10 m, m/s", ".1f"), ("sai", "Stubble smoke arriving (index)", ".2f")):
            st.altair_chart(
                alt.Chart(drivers).mark_line(strokeWidth=1.6, color="#4b5560").encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount="day")),
                    y=alt.Y(f"{col}:Q", title=None, axis=alt.Axis(tickCount=3, format=fmt)),
                    tooltip=[alt.Tooltip("time:T", format="%d %b %H:%M"), alt.Tooltip(f"{col}:Q", format=fmt, title=title)],
                ).properties(title=alt.TitleParams(title, fontSize=12, anchor="start")),
                width="stretch",
                height=150,
            )

    st.markdown("#### Station detail")
    names = list(st_df["name"])
    default = names.index(next((n for n in names if "Anand Vihar, New Delhi" in n), names[0]))
    name = st.selectbox("Station", names, index=default)
    n = names.index(name)
    hist = history_frame(res, n)
    fc = pd.DataFrame({"time": to_ist(res.fc_times), "q10": q[:, n, 0], "q50": q[:, n, 1], "q90": q[:, n, 2]})
    last = res.stations.loc[n, "last_obs"]
    st.altair_chart(fan_chart(hist, fc, f"{name} — last reading {to_ist(last):%d %b %H:%M} IST" if pd.notna(last) else name), width="stretch", height=380)

    table = st_df[["name", "day1", "day2", "day3"]].copy()
    for k in (1, 2, 3):
        table[f"day{k}_category"] = [naqi(v)[0] for v in table[f"day{k}"]]
    table = table.rename(columns={"name": "Station", "day1": "Day 1 µg/m³", "day2": "Day 2 µg/m³", "day3": "Day 3 µg/m³",
                                  "day1_category": "Day 1", "day2_category": "Day 2", "day3_category": "Day 3"})
    st.dataframe(table.round(0), hide_index=True, width="stretch")
    hourly = pd.DataFrame({"time_ist": np.repeat(to_ist(res.fc_times), len(names)), "station": np.tile(names, len(res.fc_times)),
                           "pm25_q10": q[..., 0].ravel(), "pm25_median": q[..., 1].ravel(), "pm25_q90": q[..., 2].ravel()})
    st.download_button("Download hourly forecast (CSV)", hourly.round(1).to_csv(index=False), "delhi_pm25_forecast.csv", "text/csv")


def page_backtest():
    st.subheader("How the models did on held-out periods")
    folds = {"winter": "Oct–Dec 2025 · stubble season and winter peak (held out)", "main": "Jul–Oct 2026 · monsoon to post-monsoon"}
    fold = st.radio("Test period", list(folds), format_func=folds.get, horizontal=True, key="bt_fold")
    m = load_metrics(fold)
    rows = []
    for k, v in m.items():
        if k.startswith("_") or "overall" not in v:
            continue
        sk = v.get("skill_vs_persistence", {})
        rows.append({"Model": MODEL_LABELS.get(k, k), "MAE": v["overall"]["mae"], "MAE @24h": v["h24"]["mae"], "MAE @48h": v["h48"]["mae"], "MAE @72h": v["h72"]["mae"],
                     "Day-1 NAQI accuracy": v["day1"]["category_acc"] * 100, "Day-1 skill vs persistence %": sk.get("day1", 0) * 100,
                     "Mean bias": v["overall"]["bias"]})
    if rows:
        df = pd.DataFrame(rows).sort_values("MAE")
        num = {c: st.column_config.NumberColumn(format="%.1f") for c in df.columns if c != "Model"}
        st.dataframe(df, hide_index=True, width="stretch", column_config=num)
        st.caption("Errors in µg/m³. \"Level-anchored\" models (no suffix) predict the change from the current level; \"(level)\" models predict the level directly.")

    bt = load_backtest()
    origins = pd.to_datetime(bt[f"{fold}_origins"].astype("datetime64[h]"))
    obs = bt[f"{fold}_obs"].astype(np.float32)
    obs_t0 = pd.Timestamp(int(bt[f"{fold}_obs_start"]) * 3_600_000_000_000)
    obs_times = pd.date_range(obs_t0, periods=len(obs), freq="h")
    names = list(bt["station_name"])
    c1, c2 = st.columns([2, 3])
    name = c1.selectbox("Station", names, index=names.index(next((n for n in names if "Anand Vihar, New Delhi" in n), names[0])), key="bt_station")
    n = names.index(name)
    with np.errstate(all="ignore"):  # all-missing windows are expected and skipped by nanargmax
        windows = np.stack([np.nanmean(obs[obs_times.get_loc(o) + 1 : obs_times.get_loc(o) + 73]) for o in origins])
    worst = int(np.nanargmax(windows))
    pick = c2.select_slider("Forecast issued (IST)", options=list(range(len(origins))), value=worst,
                            format_func=lambda i: f"{to_ist(origins[i]):%d %b %Y %H:%M}")
    model = st.radio("Model", ["ensemble_anchored", "ensemble"], format_func=MODEL_LABELS.get, horizontal=True, key="bt_model")
    o = origins[pick]
    i0 = obs_times.get_loc(o)
    q = bt[f"{fold}_{model}"][pick, :, n].astype(np.float32)
    hist = pd.DataFrame({"time": to_ist(obs_times[i0 - 72 : i0 + 73]), "pm25": obs[i0 - 72 : i0 + 73, n]})
    hist["kind"] = np.where(np.arange(len(hist)) <= 72, "observed", "observed (after issue)")
    fc_t = to_ist(pd.date_range(o + pd.Timedelta(hours=1), periods=72, freq="h"))
    fc = pd.DataFrame({"time": fc_t, "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]})
    per = pd.DataFrame({"time": fc_t, "pm25": bt[f"{fold}_persistence"][pick, :, n].astype(np.float32)})
    chart = fan_chart(hist, fc, f"{name}: forecast vs what happened") + alt.Chart(per).mark_line(color="#9e9e9e", strokeDash=[2, 2]).encode(x="time:T", y="pm25:Q")
    st.altair_chart(chart, width="stretch", height=380)
    st.caption("Grey dotted line: persistence (last value held flat). The slider starts at the most polluted forecast window in this period.")

    shown = [k for k in MODEL_COLORS if k in m]
    mae = pd.DataFrame({MODEL_LABELS[k]: m[k]["by_horizon"]["mae"] for k in shown})
    if len(mae.columns):
        mae["lead (h)"] = np.arange(1, len(mae) + 1)
        color = alt.Color("model:N", scale=alt.Scale(domain=[MODEL_LABELS[k] for k in shown], range=[MODEL_COLORS[k] for k in shown]),
                          legend=alt.Legend(orient="bottom", columns=3, title=None, labelLimit=260))
        st.altair_chart(
            alt.Chart(mae.melt("lead (h)", var_name="model", value_name="MAE")).mark_line(strokeWidth=2).encode(
                x=alt.X("lead (h):Q", scale=alt.Scale(domain=[1, 72], nice=False), axis=alt.Axis(values=[1, 12, 24, 36, 48, 60, 72])),
                y=alt.Y("MAE:Q", title="MAE (µg/m³)"), color=color,
                tooltip=["model:N", "lead (h):Q", alt.Tooltip("MAE:Q", format=".1f")],
            ).properties(title="Hourly error by lead time, all stations"),
            width="stretch",
            height=360,
        )


def page_about():
    st.subheader("About")
    st.markdown(
        """
Forecasts hourly **PM2.5 24–72 hours ahead** for 56 regulatory monitoring stations in Delhi, Gurugram, Noida,
Ghaziabad, Faridabad and nearby NCR towns.

**Data, all public**
- CPCB / DPCC / HSPCB / UPPCB / IMD / IITM stations via the OpenAQ archive on AWS
- NOAA GFS 1° weather forecasts (wind, boundary-layer height, inversion, rain, humidity)
- VIIRS 375 m active-fire detections (NOAA Enterprise Fire product, same algorithm family as NASA FIRMS)

**Smoke-arrival index:** each fire pixel is moved toward Delhi with the 850 hPa wind. Smoke already on its way is
known when the forecast is issued, so it is used as an input for the hours ahead.

**Models:** LightGBM, a Temporal Fusion Transformer and a wind-aware spatio-temporal graph network. Each predicts the
change from the current level ("level-anchored"), which held up best when tested on the unseen 2025 winter.
On held-out data the ensemble cut error by 25–28% compared with persistence.

**Limits:** every model underpredicts the most extreme 600–700 µg/m³ night spikes. Intervals cover about 65–73%
of hours, not the nominal 80%. Forecasts are research output, not official advisories; check CPCB's
[SAMEER](https://app.cpcbccr.com/AQI_India/) and [IITM's AQEWS](https://ews.tropmet.res.in/) for official forecasts.

Code: [github.com/akhilnathreddy/delhi-aqi-forecast](https://github.com/akhilnathreddy/delhi-aqi-forecast)
"""
    )
    fig = ROOT / "reports" / "figures" / "overview.png"
    if fig.exists():
        st.image(str(fig), caption="Network-mean PM2.5 and fire detections, Feb 2025 – Oct 2026")


st.title("Delhi-NCR PM2.5 forecast")
tab_live, tab_bt, tab_about = st.tabs(["Live forecast", "Backtest explorer", "About"])
with tab_live:
    page_live()
with tab_bt:
    page_backtest()
with tab_about:
    page_about()
