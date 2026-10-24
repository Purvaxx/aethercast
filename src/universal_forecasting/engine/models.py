"""Model adapters for the engine. Each adapter wraps code that already exists in the repository (baselines.py,
features/calendar.py, preprocessing.py) behind one interface:

    eligible(y_train, m, H) -> (bool, reason)
    fit(y, ts, train_end, m, H)            parameters from y[0..train_end] ONLY
    paths(y, ts, P, H) -> [len(P), H]      forecast for steps 1..H from each origin position p in P, reading y[0..p] only

`y` is a regular-grid array with NaN for missing points. No adapter fills missing values; "last observed value" means
the most recent non-missing observation at or before the origin.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import baselines as B
from ..features.calendar import calendar_frame
from ..preprocessing import TrainOnlyScaler

ETS_MAX_FIT = 20_000          # longest training window used to estimate ETS parameters (cost bound; documented)
RIDGE_ALPHA = 1.0
RIDGE_BASE_LAGS = (0, 1, 2, 3, 6, 12)
RIDGE_MIN_SAMPLES = 50


def last_obs_index(y: np.ndarray) -> np.ndarray:
    """For each position, the index of the most recent non-missing observation at or before it (-1 if none)."""
    idx = np.where(np.isfinite(y), np.arange(len(y)), -1)
    return np.maximum.accumulate(idx)


class Naive:
    name = "naive"

    def eligible(self, y_train, m, H):
        return (np.isfinite(y_train).any(), "no non-missing observation")

    def fit(self, y, ts, train_end, m, H):
        return self

    def paths(self, y, ts, P, H):
        lo = last_obs_index(y)[P]
        v = np.where(lo >= 0, y[np.clip(lo, 0, None)], np.nan)
        return np.repeat(v[:, None], H, axis=1)


class SeasonalNaive:
    name = "seasonal_naive"

    def eligible(self, y_train, m, H):
        if not m:
            return False, "no seasonal period for this frequency"
        return (np.isfinite(y_train).sum() >= m, f"fewer than one seasonal period ({m}) of observations")

    def fit(self, y, ts, train_end, m, H):
        self.m = m
        return self

    def paths(self, y, ts, P, H):
        out = np.empty((len(P), H))
        for k in range(1, H + 1):
            idx = P + k - self.m * math.ceil(k / self.m)
            out[:, k - 1] = np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan)
        return out


class Drift:
    name = "drift"

    def eligible(self, y_train, m, H):
        return (np.isfinite(y_train).sum() >= 2, "fewer than 2 observations")

    def fit(self, y, ts, train_end, m, H):
        return self

    def paths(self, y, ts, P, H):
        nn = np.where(np.isfinite(y))[0]
        first = nn[0] if len(nn) else 0
        lo = last_obs_index(y)[P]
        span = lo - first
        ok = (lo >= 0) & (span >= 1)
        yl = y[np.clip(lo, 0, None)]
        slope = np.where(ok, (yl - y[first]) / np.where(ok, span, 1), np.nan)
        k = np.arange(1, H + 1)[None, :] + (P - lo)[:, None]       # steps counted from the last observation
        return yl[:, None] + slope[:, None] * k


class ETS:
    name = "ets"

    def eligible(self, y_train, m, H):
        return (np.isfinite(y_train).sum() >= 30, "fewer than 30 observations")

    def fit(self, y, ts, train_end, m, H):
        self.fs = max(0, train_end + 1 - ETS_MAX_FIT)
        yf = y[self.fs: train_end + 1]
        self.m = m if (m and np.isfinite(yf).sum() >= 3 * m) else None
        self.fit_ = B.fit_hw(yf, self.m) if self.m else B.fit_ses(yf)
        if self.fit_.status != B.OK:
            raise RuntimeError(f"ETS {self.fit_.status}: {self.fit_.message}")
        if self.m:
            self.l0, self.s0 = B.hw_init(yf, self.m)
        else:
            self.l0 = float(yf[np.isfinite(yf)][0])
        self.form = self.fit_.model
        return self

    def paths(self, y, ts, P, H):
        ys = y[self.fs:]
        Q = P - self.fs
        if (Q < 0).any():
            raise ValueError("origin before the ETS fit window")
        if self.m:
            lv, sv, _ = B.hw_filter(ys, self.fit_.alpha, self.fit_.gamma, self.m, self.l0, self.s0)
            out = np.empty((len(P), H))
            for k in range(1, H + 1):
                out[:, k - 1] = lv[Q] + sv[Q + k - self.m * math.ceil(k / self.m)]
            return out
        lv, _ = B.ses_filter(ys, self.fit_.alpha, self.l0)
        return np.repeat(lv[Q][:, None], H, axis=1)


def calendar_fields(freq: str) -> tuple:
    off = pd.tseries.frequencies.to_offset(freq)
    try:
        sub_daily = pd.Timedelta(off) < pd.Timedelta("1D")
    except (ValueError, TypeError):
        sub_daily = False
    if sub_daily:
        return ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
    if freq.lstrip("0123456789").startswith(("M", "Q", "Y", "A")):
        return ("month_sin", "month_cos")
    return ("dow_sin", "dow_cos", "doy_sin", "doy_cos")


class Ridge:
    """Direct multi-step ridge: one model per step k on [target lags at the origin, trailing mean, seasonal anchor
    y_{t+k-m*ceil(k/m)}, calendar of the TARGET time t+k (known in advance)]. Scaler and coefficients from training rows only."""
    name = "ridge"

    def eligible(self, y_train, m, H):
        need = max(RIDGE_BASE_LAGS) + (m or 0) + H + RIDGE_MIN_SAMPLES
        return (np.isfinite(y_train).sum() >= need, f"fewer than {need} observations")

    def _features(self, y, ts, P, k):
        cols = [np.where(P - l >= 0, y[np.clip(P - l, 0, None)], np.nan) for l in self.lags]
        w = self.roll
        c = np.r_[0.0, np.cumsum(np.nan_to_num(y))]
        bad = np.r_[0, np.cumsum(~np.isfinite(y))]
        a = P - w + 1
        mean = (c[P + 1] - c[np.clip(a, 0, None)]) / w
        mean[(a < 0) | ((bad[P + 1] - bad[np.clip(a, 0, None)]) > 0)] = np.nan
        cols.append(mean)
        if self.m:
            idx = P + k - self.m * math.ceil(k / self.m)
            cols.append(np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan))
        tt = pd.Series(pd.DatetimeIndex(ts[P]) + k * self.step)
        cal = calendar_frame(tt, self.cal).to_numpy()
        return np.column_stack(cols + [cal])

    def fit(self, y, ts, train_end, m, H):
        self.m, self.H = m, H
        self.step = pd.tseries.frequencies.to_offset(self.freq)
        self.lags = tuple(l for l in RIDGE_BASE_LAGS)
        self.roll = m if m else 7
        self.cal = calendar_fields(self.freq)
        self.models = []
        for k in range(1, H + 1):
            P = np.arange(max(max(self.lags), self.roll - 1, (m or 0)), train_end - k + 1)     # label position p+k <= train_end
            if len(P) < RIDGE_MIN_SAMPLES:
                raise RuntimeError(f"ridge: {len(P)} training samples for step {k}")
            X, t = self._features(y, ts, P, k), y[P + k]
            ok = np.isfinite(X).all(axis=1) & np.isfinite(t)
            if ok.sum() < RIDGE_MIN_SAMPLES:
                raise RuntimeError(f"ridge: {int(ok.sum())} complete training samples for step {k}")
            sc = TrainOnlyScaler().fit(X[ok], np.ones(int(ok.sum()), bool), [f"f{i}" for i in range(X.shape[1])], "engine training rows")
            rf = B.ridge_fit(sc.transform(X[ok]), t[ok], RIDGE_ALPHA, min_per_feature=1)
            if rf.status != B.OK:
                raise RuntimeError(f"ridge fit {rf.status}")
            self.models.append((sc, rf))
        return self

    def paths(self, y, ts, P, H):
        out = np.full((len(P), H), np.nan)
        for k, (sc, rf) in enumerate(self.models, 1):
            X = self._features(y, ts, P, k)
            ok = np.isfinite(X).all(axis=1)
            if ok.any():
                out[ok, k - 1] = B.ridge_predict(rf, sc.transform(X[ok]))[0]
        return out


def build(name: str, freq: str):
    m = {"naive": Naive, "seasonal_naive": SeasonalNaive, "drift": Drift, "ets": ETS, "ridge": Ridge}[name]()
    m.freq = freq
    return m
