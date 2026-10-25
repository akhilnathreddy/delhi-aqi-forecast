"""Reference forecasts every learned model has to beat.

* **Persistence** — the last observed value, held flat for all 72 h.
* **Diurnal persistence** — the value observed at the same clock hour on the
  most recent day before the origin (captures the strong daily cycle).
* **Climatology** — training-period mean per station × month × hour of day.
* **LightGBM** — a direct multi-horizon gradient-boosted model: one model with
  the lead time as a feature, trained on randomly sampled leads, one booster
  per quantile.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..dataset import ForecastData

log = logging.getLogger(__name__)


def _last_obs(d: ForecastData, t: int) -> np.ndarray:
    """Last non-missing PM2.5 per station at or before ``t`` (within 24 h)."""
    win = d.y[t - 23 : t + 1]
    ok = np.isfinite(win)
    last = np.where(ok.any(0), win[::-1][np.argmax(ok[::-1], 0), np.arange(d.N)], np.nan)
    return np.where(np.isnan(last), np.expm1(d.logy_filled[t]), last)


def persistence(d: ForecastData, origins: np.ndarray) -> np.ndarray:
    return np.stack([np.broadcast_to(_last_obs(d, t), (d.H, d.N)) for t in origins])[..., None]


def diurnal_persistence(d: ForecastData, origins: np.ndarray) -> np.ndarray:
    filled = np.expm1(d.logy_filled)
    out = np.empty((len(origins), d.H, d.N), np.float32)
    h = np.arange(1, d.H + 1)
    back = 24 * np.ceil(h / 24).astype(int)  # same clock hour, latest day ≤ origin
    for i, t in enumerate(origins):
        out[i] = filled[t + h - back]
    return out[..., None]


class Climatology:
    def fit(self, d: ForecastData, train_idx: np.ndarray):
        ist = d.times + pd.Timedelta(hours=5, minutes=30)
        self.month, self.hour = ist.month.to_numpy(), ist.hour.to_numpy()
        logy = np.log1p(d.y)
        # Only hours that are targets of training origins (keeps blocked folds leak-free).
        hours = np.unique((train_idx[:, None] + np.arange(1, d.H + 1)[None]).ravel())
        self.table = np.full((13, 24, d.N), np.nan, np.float32)
        for m in range(1, 13):
            for hr in range(24):
                sel = hours[(self.month[hours] == m) & (self.hour[hours] == hr)]
                if len(sel):
                    self.table[m, hr] = np.nanmean(logy[sel], 0)
        # Months never seen in training fall back to the station's diurnal mean.
        fallback = np.nanmean(self.table, 0)
        self.table = np.where(np.isnan(self.table), fallback[None], self.table)
        return self

    def predict(self, d: ForecastData, origins: np.ndarray) -> np.ndarray:
        idx = origins[:, None] + np.arange(1, d.H + 1)[None]
        return np.expm1(self.table[self.month[idx], self.hour[idx]])[..., None]


# --------------------------------------------------------------------------- LightGBM
def lgbm_features(d: ForecastData, origins: np.ndarray, horizons: np.ndarray | None = None, rng=None, per_row: int = 8):
    """Tabular rows ``(origin, station, lead)``.

    ``horizons=None`` samples ``per_row`` random leads per origin-station
    (training); otherwise every listed lead is produced (inference).
    Returns ``X (DataFrame), y (log1p target), keys (origin_i, h, n)``.
    """
    ly = d.logy_filled
    N, H = d.N, d.H
    net_mean = ly.mean(1)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(ly, 0)])
    rows, keys, ys = [], [], []
    for oi, t in enumerate(origins):
        hs_all = np.arange(1, H + 1)
        if horizons is None:
            hs = np.stack([rng.choice(hs_all, per_row, replace=False) for _ in range(N)])
        else:
            hs = np.broadcast_to(np.asarray(horizons), (N, len(horizons)))
        d_i, l0 = d.cycle_for[t], d.lead0[t]
        fc = d.wx_fc[d_i, l0 + 1 : l0 + 1 + H]  # [H, N, Fw]
        fc_cum = np.nancumsum(np.nan_to_num(fc), 0) / np.arange(1, H + 1)[:, None, None]
        base = {
            "y0": ly[t], "y1": ly[t - 1], "y2": ly[t - 2], "y3": ly[t - 3], "y6": ly[t - 6], "y12": ly[t - 12],
            "y24": ly[t - 24], "y48": ly[t - 48], "y168": ly[t - 168],
            "m6": (cs[t + 1] - cs[t - 5]) / 6, "m24": (cs[t + 1] - cs[t - 23]) / 24, "m72": (cs[t + 1] - cs[t - 71]) / 72,
            "m168": (cs[t + 1] - cs[t - 167]) / 168,
            "net0": np.full(N, net_mean[t]), "net24": np.full(N, net_mean[t - 23 : t + 1].mean()),
            "obs_frac24": d.mask[t - 23 : t + 1].mean(0),
            "lat": d.lat, "lon": d.lon, "net": d.net, "station": np.arange(N),
        }
        for k, name in enumerate(["pm10", "no2", "co", "o3"]):
            base[f"{name}_0"] = d.pol[t, :, k]
        for k, name in enumerate(d.wx_names):
            base[f"an_{name}"] = d.wx_an[t, :, k]
        for k, name in enumerate(d.fire_names):
            base[name] = np.full(N, d.fire_act[t, k])
        base["sai_now"] = np.full(N, d.sai[t, 0])
        for j in range(hs.shape[1]):
            h = hs[:, j]
            r = dict(base)
            r["h"] = h
            for k, name in enumerate(d.wx_names):
                r[f"fc_{name}"] = fc[h - 1, np.arange(N), k]
            for name in ("blh", "ws10", "rain_mmh", "log_vent"):
                k = d.wx_names.index(name)
                r[f"fcmean_{name}"] = fc_cum[h - 1, np.arange(N), k]
            r["sai_h"] = d.sai[t, h]
            r["sai_max"] = np.array([d.sai[t, 1 : hh + 1].max() for hh in h])
            # Same clock hour on the most recent day before the origin.
            back = 24 * np.ceil(h / 24).astype(int)
            r["y_diurnal"] = ly[t + h - back, np.arange(N)]
            for k, name in enumerate(d.cal_names):
                r[f"cal_{name}"] = d.cal[t + h, k]
            rows.append(pd.DataFrame(r))
            keys.append(np.column_stack([np.full(N, oi), h, np.arange(N)]))
            ys.append(np.log1p(d.y[t + h, np.arange(N)]))
    X = pd.concat(rows, ignore_index=True)
    X["station"] = X["station"].astype("category")
    X["net"] = X["net"].astype("category")
    return X, np.concatenate(ys), np.concatenate(keys)


class LightGBMForecaster:
    """``residual=True`` learns ``log PM2.5(t+h) − log 6-h mean at t`` instead of the level.

    Trees cannot extrapolate beyond the target range seen in training; anchoring
    on the current level lets forecasts follow an unprecedented episode.
    """

    def __init__(self, quantiles, drop: tuple[str, ...] = (), seed: int = 42, n_estimators: int = 2000, residual: bool = False):
        self.quantiles, self.drop, self.seed, self.n_estimators = quantiles, drop, seed, n_estimators
        self.residual = residual

    def _anchor(self, X: pd.DataFrame) -> np.ndarray:
        return X["m6"].to_numpy() if self.residual else 0.0

    def _cols(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[[c for c in X.columns if not any(c.startswith(p) for p in self.drop)]]

    def fit(self, d: ForecastData, train_origins, val_origins):
        import lightgbm as lgb

        rng = np.random.default_rng(self.seed)
        Xtr, ytr, _ = lgbm_features(d, train_origins, rng=rng)
        Xva, yva, _ = lgbm_features(d, val_origins, rng=rng, per_row=4)
        ytr, yva = ytr - self._anchor(Xtr), yva - self._anchor(Xva)
        mtr, mva = np.isfinite(ytr), np.isfinite(yva)
        Xtr, ytr, Xva, yva = self._cols(Xtr)[mtr], ytr[mtr], self._cols(Xva)[mva], yva[mva]
        log.info("LightGBM: %d train rows, %d val rows, %d features", len(Xtr), len(Xva), Xtr.shape[1])
        self.models = {}
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=self.n_estimators, learning_rate=0.05, num_leaves=127,
                min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=self.seed, verbose=-1, n_jobs=4,
            )
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
            log.info("  q=%.2f best_iter=%d", q, m.best_iteration_)
            self.models[q] = m
        self.feature_names = list(Xtr.columns)
        return self

    def predict(self, d: ForecastData, origins: np.ndarray, chunk: int = 64) -> np.ndarray:
        out = np.full((len(origins), d.H, d.N, len(self.quantiles)), np.nan, np.float32)
        for s in range(0, len(origins), chunk):
            o = origins[s : s + chunk]
            X, _, keys = lgbm_features(d, o, horizons=np.arange(1, d.H + 1))
            anchor = self._anchor(X)
            X = self._cols(X)
            for k, q in enumerate(self.quantiles):
                out[s + keys[:, 0], keys[:, 1] - 1, keys[:, 2], k] = np.expm1(self.models[q].predict(X) + anchor)
        out.sort(-1)  # enforce non-crossing quantiles
        return out

    def save(self, out_dir) -> list[str]:
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for q, m in self.models.items():
            f = out_dir / f"lightgbm_q{int(round(q * 100)):02d}.txt"
            m.booster_.save_model(str(f), num_iteration=m.best_iteration_ or -1)
            files.append(f.name)
        return files

    @classmethod
    def load(cls, in_dir, quantiles, drop=(), residual=True) -> LightGBMForecaster:
        """Load saved boosters; ``predict`` then works exactly as after ``fit``."""
        from pathlib import Path

        import lightgbm as lgb

        self = cls(quantiles, drop=drop, residual=residual)
        self.models = {q: _Booster(lgb.Booster(model_file=str(Path(in_dir) / f"lightgbm_q{int(round(q * 100)):02d}.txt"))) for q in quantiles}
        self.feature_names = self.models[quantiles[0]].booster_.feature_name()
        return self

    def importance(self) -> pd.Series:
        m = self.models[0.5]
        return pd.Series(m.booster_.feature_importance("gain"), index=self.feature_names).sort_values(ascending=False)


class _Booster:
    """Adapter giving a loaded ``lgb.Booster`` the ``LGBMRegressor`` surface used above."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X):
        return self.booster_.predict(X)
