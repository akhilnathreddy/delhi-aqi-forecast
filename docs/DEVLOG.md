# Development log

How this project was built: the problems I ran into, how I diagnosed them, and what I changed. Each entry gives the symptom, the cause and the fix, with the numbers that came out of it.

---

## 1. Getting the data

### 1.1 CPCB has no bulk download API
**Symptom.** The CPCB CCR portal serves station data through an interactive web UI only. Scraping 50+ stations over 20 months that way isn't practical or reliable.

**Fix.** OpenAQ republishes every CPCB, DPCC, HSPCB, UPPCB, IMD and IITM continuous station in a public AWS bucket (`openaq-data-archive`), one gzipped CSV per station-day. There's no station index for the bucket, so I listed all **56,143 location IDs** in it, read one file from each to get its name and coordinates, and kept the ones inside the NCR box that report PM2.5. Removing low-cost sensors (PurpleAir-style `um003` channels) left **100 regulatory stations** (`configs/stations.csv`). The scan took about 20 minutes and only needs to run once (`aqf discover-stations`).

**Gotcha.** The current CPCB feeds in the archive start on **2025-02-18**. Older years exist under the same IDs but with a multi-year gap, so the study period is Feb 2025 → Oct 2026.

### 1.2 Timestamps and hour labels
**Symptom.** Readings are stamped in IST at the *end* of each 15-minute interval (`10:00+05:30` covers 09:45–10:00).

**Fix.** Convert to UTC, subtract one second, then floor to the hour. Every hour is therefore labelled by its start, in UTC. An hour is kept only if at least 2 of its 4 slots are present. Values outside physical range are dropped (PM2.5 ≤ 0 or > 999 µg/m³, where CPCB analysers saturate), and runs of ≥ 6 identical hourly values are masked as a stuck analyser. A unit test pins this behaviour down. My first version of that test was itself wrong about which hour a reading belongs to, which is a good reason to have the test.

**Coverage.** 56 of the 100 stations have ≥ 60 % valid hourly PM2.5 over the period. Overall, 82 % of station-hours are observed.

### 1.3 The obvious weather and fire APIs were unreachable
**Symptom.** Open-Meteo, the NASA FIRMS API, the OpenAQ API and data.gov.in were all blocked from the machine I built and trained on. Only package registries, GitHub and AWS/GCS public buckets were reachable.

**Fix.** I moved everything onto AWS Open Data:
- **Weather:** NOAA GFS forecasts (`noaa-gfs-bdp-pds`). This is actually *better* for forecasting than reanalysis or observed weather, because it is the information a forecaster really has at issue time (see 3.1).
- **Fires:** NOAA's Enterprise Active Fire EDR for VIIRS (`noaa-nesdis-{snpp,n20}-pds/VIIRS_EFIRE_VIIRSI_EDR`). It is the same 375 m I-band algorithm family that NASA FIRMS serves. A FIRMS-API downloader with the same output schema is included for anyone with a `FIRMS_MAP_KEY`.

### 1.4 GFS files are 45 MB each
**Symptom.** 662 days × 35 forecast leads × 45 MB is about 1 TB.

**Fix.** Every GRIB2 file has a `.idx` sidecar listing each field's byte offset. I parse it, request only the 10 fields I need with HTTP **range requests** (merging nearby ranges), decode them with ecCodes and crop to a 1° box over NCR and the Punjab source region. That's about **0.7 MB per lead instead of 45 MB**. Averaged and accumulated variants (`0-6 hour ave`) are skipped so only instantaneous fields are used.

### 1.5 VIIRS fire granules: about 1000 files per satellite per day
**Symptom.** Each granule is ~1 MB and covers 85 seconds of orbit anywhere on Earth. Downloading every granule for 23 months × 2 satellites wasn't an option.

**Fix.** Two filters:
1. **Time window.** S-NPP and NOAA-20 are sun-synchronous (13:30 / 01:30 local solar time), so the region can only be seen within ±75 min of those times at 75°E.
2. **Footprint.** Each netCDF stores a `geospatial_bounds` WKT polygon in its header. I fetch only the **first 24 kB**, regex out the polygon, and download the full granule only if the polygon overlaps the fire box.

In January 2025, for example, **6,487 granules were checked and 366 downloaded** (6 %). The full 23-month, two-satellite archive (562k fire pixels) came down in about half an hour.

### 1.6 Night passes lost at month boundaries
**Symptom.** While reviewing the fire downloader I noticed a gap. The night pass for day *D* happens around 20:30 UTC on day *D−1*. Fires were stored per month and filtered to that month's dates, so the night pass on the last day of each month was dropped from both months.

**Fix.** For each UTC day, compute the overpass windows for *D* and *D+1* and keep only granules that *start* on *D*. Every granule now belongs to exactly one day.

---

## 2. Engineering problems

### 2.1 Segfault (exit code 139) in the fire downloader
**Symptom.** The fire download crashed with a segmentation fault after a few minutes.

**Cause.** I was decoding netCDF granules in 16–24 threads, and libhdf5 (under netCDF4) is **not thread-safe**.

**Fix.** A lock around decoding only. Downloads stay parallel; decoding takes milliseconds, so serialising it costs almost nothing. I put the same guard around ecCodes GRIB decoding.

### 2.2 Dropped connections and slow small requests
**Symptom.** `http.client.IncompleteRead` killed long runs. GFS downloads crawled at about **6 days per minute** even though bandwidth was fine.

**Cause.** Each range request opened a fresh TLS connection. With tens of thousands of small requests, the handshakes dominated, and the occasional dropped connection wasn't retried.

**Fix.** One keep-alive `requests.Session` per worker thread, exponential-backoff retries on any request exception, and a fresh session after a failure. Single-request latency went from ~0.5–1.5 s to **~0.2 s**, and GFS throughput rose to about **18 days per minute**. Downloads became resumable: each file is skipped if it already exists.

### 2.3 Training a Transformer and a GNN on 4 CPU cores
**Symptom.** A full TFT epoch, with every station at every 6-hourly origin, was projected at ~16 minutes. That would put 15 epochs × 2 folds × 2 variants at around 16 hours.

**Fix.**
- **TFT:** each batch takes 8 origins and a **random 24 of the 56 stations** per origin. Every origin is still seen each epoch, and the per-station view is regularised. Epoch time dropped to **~4.6 min**.
- **ST-GNN:** one sample is a whole graph, so it uses 3-hourly origins to get enough samples. Epochs take ~3.3 min.
- Both have an 80-minute time budget per run on top of early stopping (see 5).

### 2.4 Small bugs caught by a smoke run
Before the full run I trained every model for one epoch on a 2-month slice. That caught a `ValueError: Object arrays cannot be loaded when allow_pickle=False` (feature-name arrays were saved as Python objects; now saved as `str`) before it could cost an hour of training.

---

## 3. Keeping the evaluation honest

### 3.1 Only use what a forecaster would actually have
- **Weather:** the future part of each sample uses the *GFS forecast* from the newest 00Z cycle **published** by the origin hour (5 h latency), never observed future weather. Leads up to 102 h cover origins issued any time during the day.
- **Fires:** a detection only counts **3 h after** the satellite pass (FIRMS near-real-time latency). A unit test checks that the smoke index for every lead is exactly zero before detection + latency.
- **Missing targets** are masked out of the loss and the metrics rather than imputed.

### 3.2 Blocked folds with an embargo
Random splits leak badly in hourly series. I use blocked chronological partitions, keep an origin only if its whole 72 h target window lies inside its partition, and skip the first 72 h after every boundary. No target hour is ever shared between train and test.

### 3.3 A leak I almost shipped: climatology
When I added the held-out `winter` fold, whose training data lies on both sides of the test season, I noticed the climatology baseline was fitted on the time span from the first to the last training origin. That span included the test season. I changed it to use only hours that are targets of training origins.

---

## 4. Modelling problems

### 4.1 Every model collapsed on the held-out winter
**Symptom.** On the `winter` fold (test Oct–Dec 2025), the first LightGBM, TFT and ST-GNN were *worse than persistence* on day 1, with mean bias of **−60 to −90 µg/m³**. The ST-GNN reached MAE 93.6 against 80.8 for persistence.

**Diagnosis.** That fold's training data has no November–December. The highest training targets are well below the 250–700 µg/m³ of the winter peak, and the validation months (Aug–Sep) are the cleanest of the year, so early stopping also preferred "low" models. Models that predict the *level* can't go above what they have seen; trees literally can't extrapolate.

**Fix: level anchoring.** Each model predicts the *change* in log-PM2.5 from the last-6-hour mean instead of the level. The anchor carries the current regime, and the model only learns how the next 72 h deviate from it.

| Winter fold | Before (level) | After (anchored) | Bias before → after |
|---|---|---|---|
| LightGBM | 71.8 | 65.6 | −62.5 → −52.7 |
| TFT | 71.2 | 62.2 | −60.7 → −42.3 |
| ST-GNN | 93.6 | **61.9** | −89.1 → **−23.9** |
| Ensemble | 78.3 | **60.2** | −72.5 → −41.4 |

On the forward-in-time `main` fold, anchoring cost little (ensemble 16.3 → 16.7).

**Caveat, stated plainly.** I added anchoring *after* seeing the winter failure. It's reported next to the originals, not instead of them, and it ran unchanged on `main`.

### 4.2 Fire inputs didn't help, and why
**Symptom.** Removing the fire and smoke-index inputs left LightGBM unchanged on `main` (17.2 → 17.0) and even *better* on `winter` (71.8 → 67.1).

**Diagnosis.** Daily correlations by burning season:

| Season | Fire pixels/day | corr(PM2.5, smoke index) | corr(PM2.5, mixing height) |
|---|---|---|---|
| Wheat residue, Apr–May 2025 | 918 | +0.07 | +0.21 |
| Paddy stubble, Oct–Nov 2025 | 389 | **+0.56** | −0.58 |
| Wheat residue, Apr–May 2026 | 807 | +0.03 | +0.48 |

Spring has more than twice as many fires, but the deep afternoon mixing layer dilutes the smoke and PM2.5 barely responds. Most fire examples in the training data are therefore "fire with no effect", and with only one autumn season to learn from, the models learn to mostly ignore fire. The smoke index is physically right; the models just need more autumn seasons, or an explicit smoke × mixing-height interaction, to use it.

### 4.3 Climatology was a disaster in winter
MAE 126.7 on `winter`, because its table had no November or December entries and fell back to year-round hourly means. I kept it in the results as a reminder of why seasonal baselines need the season in their training data.

---

## 5. Training details and early stopping

All neural models use the same recipe:

| Setting | Value |
|---|---|
| Loss | pinball (quantile) loss for the 10/50/90 % quantiles, on standardised log-PM2.5, masked where targets are missing |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4, gradient clipping at norm 1.0 |
| LR schedule | ReduceLROnPlateau: halve the learning rate after 1 epoch without validation improvement |
| **Early stopping** | stop after **4 epochs** without a new best validation loss (minimum improvement 1e-4), then **restore the best epoch's weights** |
| Caps | 15 epochs for the TFT, 20 for the ST-GNN, and an 80-minute time budget |

How it played out (validation pinball loss):

| Run | Epochs run | Best epoch | Best val | Val at last epoch |
|---|---|---|---|---|
| main · TFT | 11 | 7 | 0.1519 | 0.1588 |
| main · ST-GNN | 7 | 3 | 0.1456 | 0.1544 |
| main · TFT (anchored) | 13 | 9 | 0.1508 | 0.1515 |
| main · ST-GNN (anchored) | 11 | 7 | 0.1504 | 0.1606 |
| winter · TFT | 12 | 8 | 0.1726 | 0.1832 |
| winter · ST-GNN | 6 | 2 | 0.1720 | 0.1851 |
| winter · TFT (anchored) | 11 | 7 | 0.1693 | 0.1739 |
| winter · ST-GNN (anchored) | 8 | 4 | 0.1745 | 0.1796 |
| **final · TFT (anchored)** | 7 | **3** | 0.1755 | 0.1871 |
| **final · ST-GNN (anchored)** | 8 | **4** | 0.1719 | 0.2084 |

The pattern is the same in every run. Training loss keeps falling (final TFT: 0.1865 → 0.1329 over 7 epochs), while validation loss bottoms out after a few epochs and then wanders upward. Without early stopping the saved models would be the overfitted last epoch. With it, the final TFT, for example, ran 4 more epochs after its best (epochs 4–7 never beat 0.1755) and then reverted to the **epoch 3** weights. The ST-GNN usually peaks earlier than the TFT, around epochs 2–4, because each of its samples is a whole graph and it sees more correlated data per step.

**LightGBM** uses the same idea: up to 2000 trees with learning rate 0.05, stopping when the validation pinball loss hasn't improved for 100 rounds. Across the 30 fold-run boosters (5 variants × 3 quantiles × 2 folds), the best iteration was mostly **50–200** (range 23–482). That's far below the cap and a sign of the shift between training and validation seasons. The final median model (validation in October 2026) went to 727.

### The serving models
`models/final/` is trained on **all data from Feb 2025 to Sep 2026**, with **1–20 Oct 2026** held back only for early stopping. October is the start of the stubble season, so early stopping prefers models that handle the autumn build-up.

---

## 6. Building the web app

| Problem | Cause | Fix |
|---|---|---|
| A "live" forecast from the archive ended *before* now | The public OpenAQ archive lags real time by **2–4 days** (85 h when I tested) | The model rolls forward across the gap; its median fills the missing hours, flagged as unobserved through the same mask the models were trained with. The forecast is then issued from the current hour with the newest GFS run and fires, and the filled stretch is drawn as a dotted line. With an `OPENAQ_API_KEY` the app tops up recent PM2.5 from the v3 API instead |
| Not enough history before the forecast hour | Archive lag + 7 days of encoder history > the 8-day window I first fetched | Fetch a 12-day window |
| `ValueError: assignment destination is read-only` while filling the gap | pandas 3 (copy-on-write) returns read-only NumPy views | Copy the array before writing |
| Charts drawn as flat lines | Altair parses `Wind 10 m (m/s)` as an aggregation `m(...)` because of the parentheses | Plain field names; human-readable titles set separately |
| Small charts collapsed to a few pixels | Recent Streamlit ignores a chart's own height | Pass `width="stretch"` and an explicit `height` to `st.altair_chart` |
| Observed and model-filled lines had swapped dash styles | Altair assigns dash patterns in alphabetical order | Explicit domain → dash mapping, limited to the series present |
| One failed data source could crash the app | Unhandled download errors | A failed fire download becomes a visible note ("assumes no smoke"); a total outage shows a clear message |

The app itself has a headless test (`tests/test_app.py`, Streamlit's `AppTest`), so CI catches a broken page before deployment.

---

## 7. What I'd do next
- More burning seasons, plus a **smoke × mixing-height** interaction feature (4.2).
- **Conformal calibration per season.** The nominal 80 % intervals cover 64–73 % of hours.
- Higher-resolution meteorology (ERA5 / IMD WRF) and proper back-trajectories (HYSPLIT) instead of straight-line advection.
- Automatic weekly retraining of `models/final` as new data arrives.
