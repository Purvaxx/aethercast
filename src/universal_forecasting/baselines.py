"""Baseline forecasters for the Phase 3B benchmark (protocol phase3b-protocol-v1.0).

Every function works on ONE entity's observed series `y` (numpy, NaN = missing) and integer origin positions `P`
(row index of the origin inside that series); forecasts use y[0..P] only. Failures return NaN with a status code;
nothing is ever substituted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

OK, MISSING, INSUFFICIENT, FIT_FAILED, FORECAST_FAILED, NA = "ok", "missing_input", "insufficient_history", "fit_failed", "forecast_failed", "not_applicable"


def _status(f):
    return np.where(np.isfinite(f), OK, MISSING)


# ------------------------------------------------------------------ naive / seasonal naive / drift
def naive_level(y, P, h, kind):
    """value_h and mean_h -> y_t (flat forecast)."""
    f = y[P].astype("float64")
    return f, _status(f)


def window_mean(y, a, b):
    """mean of y[a..b] inclusive per pair; NaN if any NaN or out of range."""
    c = np.r_[0.0, np.cumsum(np.nan_to_num(y))]
    k = np.r_[0, np.cumsum(np.isnan(y))]
    ok = (a >= 0) & (b < len(y)) & (a <= b)
    a2, b2 = np.clip(a, 0, len(y) - 1), np.clip(b, 0, len(y) - 1)
    s = c[b2 + 1] - c[a2]
    bad = k[b2 + 1] - k[a2]
    out = s / (b2 - a2 + 1)
    out[~ok | (bad > 0)] = np.nan
    return out


def seasonal_naive(y, P, h, m, kind):
    if h > m:
        raise ValueError("protocol requires h <= m")
    if kind == "value":
        idx = P + h - m
        f = np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan)
    elif kind == "mean":
        f = window_mean(y, P + 1 - m, P + h - m)
    else:
        raise ValueError(kind)
    return f, _status(f)


def drift(y, P, h, kind):
    """Hyndman drift: slope from the first non-NaN observation to y_t."""
    nn = np.where(~np.isnan(y))[0]
    if len(nn) == 0:
        return np.full(len(P), np.nan), np.full(len(P), INSUFFICIENT, dtype=object)
    first = nn[0]
    n = P - first
    b = np.where(n >= 1, (y[P] - y[first]) / np.where(n >= 1, n, 1), np.nan)
    if kind == "value":
        f = y[P] + h * b
    elif kind == "mean":
        f = y[P] + b * (h + 1) / 2
    elif kind == "log_return":          # y = ln p
        f = h * b
    else:
        raise ValueError(kind)
    st = np.where(n >= 1, _status(f), INSUFFICIENT)
    return f, st


def trailing_realized_vol(lnp, P, h):
    """Persistence: sqrt(sum of the last h squared log returns ending at t)."""
    r = np.r_[np.nan, np.diff(lnp)]
    c = np.r_[0.0, np.cumsum(np.nan_to_num(r) ** 2)]
    k = np.r_[0, np.cumsum(np.isnan(r))]
    a = P - h + 1
    ok = a >= 1
    a2 = np.clip(a, 0, None)
    f = np.sqrt(c[P + 1] - c[a2])
    f[~ok | ((k[P + 1] - k[a2]) > 0)] = np.nan
    return f, np.where(ok, _status(f), INSUFFICIENT)


# ------------------------------------------------------------------ ETS
@dataclass
class ETSFit:
    model: str
    alpha: float = np.nan
    gamma: float = np.nan
    m: int = 0
    sse: float = np.nan
    n_fit: int = 0
    status: str = OK
    message: str = ""
    skipped_nan: int = 0


def ses_filter(y, alpha, l0):
    """ETS(A,N,N): returns level after observing each y_t (NaN obs -> no update) and one-step SSE."""
    n = len(y)
    lv = np.empty(n)
    l = l0
    sse = 0.0
    yl = y.tolist()
    for t in range(n):
        v = yl[t]
        if v == v:
            e = v - l
            sse += e * e
            l = l + alpha * e
        lv[t] = l
    return lv, sse


def hw_filter(y, alpha, gamma, m, l0, s0):
    """ETS(A,N,A): level and seasonal state after each y_t; s[t] is the seasonal index for time t."""
    n = len(y)
    lv = np.empty(n)
    sv = np.empty(n)
    l = l0
    seas = list(s0)                      # seasonal values for the last m times (cyclic)
    sse = 0.0
    yl = y.tolist()
    for t in range(n):
        j = t % m
        s_prev = seas[j]
        v = yl[t]
        if v == v:
            e = v - l - s_prev
            sse += e * e
            l = l + alpha * e
            seas[j] = s_prev + gamma * e
        lv[t] = l
        sv[t] = seas[j]
    return lv, sv, sse


def fit_ses(y_fit, min_obs=30):
    v = y_fit[~np.isnan(y_fit)]
    if len(v) < min_obs:
        return ETSFit("ETS(A,N,N)", status=INSUFFICIENT, n_fit=len(v), message=f"{len(v)} < {min_obs} observations")
    l0 = float(v[0])
    try:
        r = minimize_scalar(lambda a: ses_filter(y_fit, a, l0)[1], bounds=(1e-4, 0.9999), method="bounded", options={"xatol": 1e-5})
    except Exception as ex:                                  # noqa: BLE001
        return ETSFit("ETS(A,N,N)", status=FIT_FAILED, n_fit=len(v), message=repr(ex))
    if not (r.success and np.isfinite(r.fun) and np.isfinite(r.x)):
        return ETSFit("ETS(A,N,N)", status=FIT_FAILED, n_fit=len(v), message=str(getattr(r, "message", "")))
    return ETSFit("ETS(A,N,N)", alpha=float(r.x), sse=float(r.fun), n_fit=len(v), message="bounded scalar search converged")


def hw_init(y_fit, m):
    first = y_fit[:m]
    l0 = float(np.nanmean(first))
    s0 = np.where(np.isnan(first), 0.0, first - l0)
    return l0, s0


def fit_hw(y_fit, m):
    v = y_fit[~np.isnan(y_fit)]
    if len(v) < 3 * m:
        return ETSFit("ETS(A,N,A)", m=m, status=INSUFFICIENT, n_fit=len(v), message=f"{len(v)} < 3m observations")
    l0, s0 = hw_init(y_fit, m)

    def sse(p):
        a, g = p
        return hw_filter(y_fit, a, g * (1 - a), m, l0, s0)[2]
    try:
        r = minimize(sse, x0=[0.1, 0.05 / 0.9], bounds=[(1e-4, 0.9999), (1e-4, 1.0)], method="L-BFGS-B")
    except Exception as ex:                                  # noqa: BLE001
        return ETSFit("ETS(A,N,A)", m=m, status=FIT_FAILED, n_fit=len(v), message=repr(ex))
    if not (r.success and np.isfinite(r.fun) and np.all(np.isfinite(r.x))):
        return ETSFit("ETS(A,N,A)", m=m, status=FIT_FAILED, n_fit=len(v), message=str(r.message))
    a, g = r.x
    return ETSFit("ETS(A,N,A)", alpha=float(a), gamma=float(g * (1 - a)), m=m, sse=float(r.fun), n_fit=len(v), message=str(r.message))


def ets_level_forecast(y, fit_start, fit_end, P, h, kind, m=None, cache: dict | None = None):
    """Fit on y[fit_start..fit_end] (training only), filter forward with fixed parameters, forecast at origins P.
    `cache` (optional dict) reuses the fit + filtered states for further targets of the same series."""
    key = ("lvl", fit_start, fit_end, m)
    if cache is not None and key in cache:
        fit, lv, sv = cache[key]
    else:
        y_fit = y[fit_start: fit_end + 1]
        fit = fit_hw(y_fit, m) if m else fit_ses(y_fit)
        lv = sv = None
        if fit.status == OK:
            ys = y[fit_start:]
            fit.skipped_nan = int(np.isnan(ys).sum())
            if m:
                l0, s0 = hw_init(y_fit, m)
                lv, sv, _ = hw_filter(ys, fit.alpha, fit.gamma, m, l0, s0)
            else:
                lv, _ = ses_filter(ys, fit.alpha, float(ys[~np.isnan(ys)][0]))
        if cache is not None:
            cache[key] = (fit, lv, sv)
    if fit.status != OK:
        return np.full(len(P), np.nan), np.full(len(P), fit.status, dtype=object), fit
    ys = y[fit_start:]
    Q = P - fit_start
    if (Q < 0).any():
        raise ValueError("origin before the ETS fit window")
    if m:
        steps = [lv[Q] + sv[Q + k - m] for k in range(1, h + 1)]   # seasonal state of time t+k-m (<= t since h <= m)
        f = steps[-1] if kind == "value" else np.mean(steps, axis=0)
    else:
        if kind in ("value", "mean"):
            f = lv[Q]
        elif kind == "log_return":           # ys = ln p
            f = lv[Q] - ys[Q]
        else:
            raise ValueError(kind)
    f = np.asarray(f, dtype="float64")
    return f, np.where(np.isfinite(f), OK, FORECAST_FAILED), fit


def ets_vol_forecast(lnp, fit_start, fit_end, P, h, cache: dict | None = None):
    """ETS(A,N,N) on squared log returns (smoothed variance), forecast sqrt(h * sigma2_t)."""
    key = ("vol", fit_start, fit_end)
    if cache is not None and key in cache:
        fit, lv = cache[key]
    else:
        r2 = np.r_[np.nan, np.diff(lnp)] ** 2
        fit = fit_ses(r2[fit_start: fit_end + 1])
        lv = None
        if fit.status == OK:
            ys = r2[fit_start:]
            nn = ys[~np.isnan(ys)]
            lv, _ = ses_filter(ys, fit.alpha, float(nn[:30].mean()))
            fit.skipped_nan = int(np.isnan(ys).sum())
        if cache is not None:
            cache[key] = (fit, lv)
    if fit.status != OK:
        return np.full(len(P), np.nan), np.full(len(P), fit.status, dtype=object), fit
    f = np.sqrt(h * np.clip(lv[P - fit_start], 0, None))
    return f, np.where(np.isfinite(f), OK, FORECAST_FAILED), fit


# ------------------------------------------------------------------ ridge
@dataclass
class RidgeFit:
    alpha: float
    coef: np.ndarray | None = None
    intercept: float = np.nan
    x_mean: np.ndarray | None = None
    n_train: int = 0
    status: str = OK


def ridge_fit(X, y, alpha=1.0, min_per_feature=10):
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(y) < min_per_feature * X.shape[1]:
        return RidgeFit(alpha, n_train=len(y), status=INSUFFICIENT)
    xm, ym = X.mean(axis=0), y.mean()
    Xc = X - xm
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    try:
        coef = np.linalg.solve(A, Xc.T @ (y - ym))
    except np.linalg.LinAlgError:
        return RidgeFit(alpha, n_train=len(y), status=FIT_FAILED)
    return RidgeFit(alpha, coef, float(ym - xm @ coef), xm, len(y))


def ridge_predict(fit: RidgeFit, X):
    if fit.status != OK:
        return np.full(len(X), np.nan), np.full(len(X), fit.status, dtype=object)
    f = np.asarray(X, dtype="float64") @ fit.coef + fit.intercept
    return f, np.where(np.isfinite(f), OK, FORECAST_FAILED)


# ------------------------------------------------------------------ metrics
def metrics(f, y, denom=None, directional=False):
    e = f - y
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(e))) if len(y) else np.nan, "rmse": float(np.sqrt(np.mean(e ** 2))) if len(y) else np.nan}
    if denom is not None:
        d = np.asarray(denom, dtype="float64")
        ok = np.isfinite(d) & (d > 0)
        out["mase"] = float(np.mean(np.abs(e[ok]) / d[ok])) if ok.any() else np.nan
        out["mase_n_excluded"] = int((~ok).sum())
    if directional:
        tie = (f == 0) | (y == 0)
        out["directional_accuracy"] = float(np.mean(np.sign(f[~tie]) == np.sign(y[~tie]))) if (~tie).any() else np.nan
        out["directional_ties"] = int(tie.sum())
    return out
