"""Dataset profiling and validation for a ForecastRequest. Separates blocking errors from warnings and returns, per
(group, target), a regular-grid series ready for the models. Nothing is filled or silently deleted: inserted grid
points stay missing and every removal (exact duplicate rows only) is reported with its count."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schemas import ForecastRequest, Issue, Severity

SUSPICIOUS_NAME = re.compile(r"(future|next|lead|ahead|forward|target|label|tomorrow)", re.I)
REGULAR_SHARE = 0.80          # share of consecutive gaps equal to the modal gap required to call a series regular
NEAR_CONSTANT_SHARE = 0.99


@dataclass
class PreparedSeries:
    group: str | None
    target: str
    timestamps: pd.DatetimeIndex          # regular grid
    y: np.ndarray                         # NaN where missing (observed missing or inserted grid point)
    frequency: str
    frequency_source: str
    seasonal_period: int | None
    seasonal_source: str
    inserted_points: int
    issues: list = field(default_factory=list)
    ok: bool = True


def _freq_alias(delta: pd.Timedelta) -> str | None:
    try:
        return pd.tseries.frequencies.to_offset(delta).freqstr
    except ValueError:
        return None


def _infer_freq(ts: pd.Series) -> tuple[str | None, float]:
    d = ts.diff().dropna()
    if d.empty:
        return None, 0.0
    modal = d.mode().iloc[0]
    share = float((d == modal).mean())
    # calendar frequencies (month/quarter starts) are not constant timedeltas: let pandas try
    try:
        inferred = pd.infer_freq(ts) if len(ts) >= 3 else None
    except (TypeError, ValueError):
        inferred = None
    if inferred and inferred.lstrip("0123456789").startswith(("M", "Q", "Y", "A")):
        return inferred, 1.0
    return _freq_alias(modal), share


SEASONAL_BY_DELTA = {pd.Timedelta("1min"): 1440, pd.Timedelta("10min"): 144, pd.Timedelta("15min"): 96, pd.Timedelta("30min"): 48, pd.Timedelta("1h"): 24,
                     pd.Timedelta("1D"): 7, pd.Timedelta("7D"): 52}


def seasonal_from_freq(freq: str | None) -> int | None:
    """Fixed rule (reported in the response): sub-daily -> one day; daily -> one week; weekly -> one year; monthly -> 12; quarterly -> 4."""
    if not freq:
        return None
    off = pd.tseries.frequencies.to_offset(freq)
    base = off.freqstr.lstrip("0123456789")
    if base == "D":
        return SEASONAL_BY_DELTA.get(pd.Timedelta(days=off.n))
    try:
        return SEASONAL_BY_DELTA.get(pd.Timedelta(off))
    except (ValueError, TypeError):
        if off.n != 1:
            return None
        return 12 if base.startswith("M") else 4 if base.startswith("Q") else 5 if base == "B" else 52 if base.startswith("W") else None


def profile(req: ForecastRequest) -> tuple[dict, list[Issue], list[PreparedSeries]]:
    issues: list[Issue] = []
    df = req.data.copy()
    tcol, gcol = req.timestamp_column, req.group_column
    prof = {"n_rows_in": int(len(df)), "columns": {}}
    # ---- timestamps
    raw_ts = df[tcol]
    ts = pd.to_datetime(raw_ts, errors="coerce")
    bad = ts.isna() & raw_ts.notna()
    nulls = raw_ts.isna()
    if bad.any() or nulls.any():
        issues.append(Issue("TIMESTAMP_PARSE_FAILED", Severity.ERROR, f"{int(bad.sum())} timestamps could not be parsed and {int(nulls.sum())} are empty",
                            "Fix or remove these rows; the engine does not guess timestamps.", details={"examples": raw_ts[bad].astype(str).head(5).tolist()}))
        return prof, issues, []
    if getattr(ts.dt, "tz", None) is not None:
        issues.append(Issue("TIMEZONE_CONVERTED", Severity.WARNING, f"timezone-aware timestamps ({ts.dt.tz}) were converted to naive UTC",
                            "Pass naive local timestamps if calendar effects should follow local time."))
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df[tcol] = ts
    # ---- column typing
    for c in df.columns:
        if c in (tcol, gcol):
            continue
        s = df[c]
        kind = "numeric" if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) else "categorical"
        vc = s.value_counts(dropna=True)
        top = float(vc.iloc[0] / max(s.notna().sum(), 1)) if len(vc) else 1.0
        tags = []
        if s.nunique(dropna=True) <= 1:
            tags.append("constant")
        elif top >= NEAR_CONSTANT_SHARE:
            tags.append("near_constant")
        prof["columns"][str(c)] = {"type": kind, "missing_fraction": float(s.isna().mean()), "tags": tags}
    for c in req.target_columns:
        info = prof["columns"][c]
        if info["type"] != "numeric":
            issues.append(Issue("TARGET_NOT_NUMERIC", Severity.ERROR, f"target '{c}' is not numeric", "Convert it to numbers or choose a numeric target.", f"column:{c}"))
        elif np.isinf(df[c].to_numpy(dtype="float64")).any():
            issues.append(Issue("TARGET_NON_FINITE", Severity.ERROR, f"target '{c}' contains infinite values", "Replace or remove them; they are not treated as missing.",
                                f"column:{c}", {"n_inf": int(np.isinf(df[c].to_numpy(dtype='float64')).sum())}))
        elif "constant" in info["tags"]:
            issues.append(Issue("TARGET_CONSTANT", Severity.WARNING, f"target '{c}' is constant", "Every model will return the constant.", f"column:{c}"))
    # ---- exact duplicate rows (only these are removed, and counted)
    dup = df.duplicated()
    if dup.any():
        issues.append(Issue("DUPLICATE_ROWS_REMOVED", Severity.WARNING, f"{int(dup.sum())} exact duplicate rows removed", "Check the export that produced the file.",
                            details={"n_removed": int(dup.sum())}))
        df = df[~dup]
    prof["n_rows_after_exact_duplicates"] = int(len(df))
    # ---- covariates / leakage screening (covariates are NOT used by 4A models)
    if req.known_future_covariates:
        issues.append(Issue("COVARIATES_NOT_USED", Severity.WARNING, f"known-future covariates {req.known_future_covariates} were accepted but are not used by Phase 4A models",
                            "Covariate-aware models are deferred; forecasts are univariate per series."))
    if any(i.severity == Severity.ERROR for i in issues):
        return prof, issues, []
    groups = [(None, df)] if gcol is None else [(str(k), x) for k, x in df.groupby(gcol, sort=True)]
    prof["n_series"] = len(groups) * len(req.target_columns)
    freqs, prepared = {}, []
    for g, x in groups:
        was_sorted = x[tcol].is_monotonic_increasing
        x = x.sort_values(tcol, kind="mergesort")
        scope = f"series:{g}"
        if not was_sorted:
            issues.append(Issue("TIMESTAMPS_UNSORTED", Severity.WARNING, f"rows of {g or 'the series'} were not in time order and were sorted", "", scope))
        dts = x[tcol].duplicated(keep=False)
        series_issues = []
        if dts.any():
            series_issues.append(Issue("DUPLICATE_TIMESTAMPS", Severity.ERROR, f"{int(x[tcol].duplicated().sum())} timestamps occur more than once with different values",
                                       "Aggregate or de-duplicate them explicitly (the engine will not choose which value is right).", scope,
                                       {"examples": x.loc[dts, tcol].astype(str).head(5).tolist()}))
        if req.frequency:
            freq, src = pd.tseries.frequencies.to_offset(req.frequency).freqstr, "declared"
            share = 1.0
        else:
            freq, share = _infer_freq(x[tcol])
            src = "inferred"
        freqs[g] = freq
        if freq is None or share < REGULAR_SHARE:
            series_issues.append(Issue("IRREGULAR_FREQUENCY", Severity.ERROR, f"timestamps of {g or 'the series'} are irregular (modal-gap share {share:.2f})",
                                       "Resample the data to a regular frequency yourself, or declare `frequency` if the irregularity is only missing points.", scope))
        for t in req.target_columns:
            ps = PreparedSeries(g, t, pd.DatetimeIndex([]), np.array([]), freq or "", src, None, "", 0, list(series_issues), not series_issues)
            if ps.ok:
                grid = pd.date_range(x[tcol].iloc[0], x[tcol].iloc[-1], freq=freq)
                off = x.set_index(tcol)[t].reindex(grid)
                n_off = int((~x[tcol].isin(grid)).sum())
                if n_off:
                    ps.issues.append(Issue("OFF_GRID_TIMESTAMPS", Severity.ERROR, f"{n_off} timestamps do not lie on the {freq} grid", "Align or resample the timestamps.", f"series:{g}|{t}"))
                    ps.ok = False
                inserted = len(grid) - len(x)
                y = off.to_numpy(dtype="float64")
                miss = float(np.isnan(y).mean())
                if ps.ok and miss > req.max_missing_fraction:
                    ps.issues.append(Issue("TOO_MANY_MISSING", Severity.ERROR, f"{miss:.1%} of grid points are missing (limit {req.max_missing_fraction:.0%})",
                                           "Supply more complete data or raise max_missing_fraction knowingly.", f"series:{g}|{t}", {"missing_fraction": miss}))
                    ps.ok = False
                elif ps.ok and (inserted > 0 or np.isnan(y).any()):
                    ps.issues.append(Issue("MISSING_POINTS", Severity.WARNING, f"{int(np.isnan(y).sum())} missing points ({inserted} absent timestamps inserted); they are left missing, never filled",
                                           "", f"series:{g}|{t}", {"missing_fraction": miss, "inserted": inserted}))
                ps.timestamps, ps.y, ps.inserted_points = grid, y, max(inserted, 0)
                if req.seasonal_period:
                    ps.seasonal_period, ps.seasonal_source = req.seasonal_period, "declared"
                else:
                    ps.seasonal_period = seasonal_from_freq(freq)
                    ps.seasonal_source = f"derived from frequency {freq}" if ps.seasonal_period else "none (no rule for this frequency)"
                n_obs = int(np.isfinite(y).sum())
                if ps.ok and n_obs < req.horizon + 1:
                    ps.issues.append(Issue("INSUFFICIENT_HISTORY", Severity.ERROR, f"{n_obs} observations cannot support a {req.horizon}-step forecast",
                                           "Provide a longer history or a shorter horizon.", f"series:{g}|{t}"))
                    ps.ok = False
                if ps.ok:
                    ps.issues += _leakage_screen(x.set_index(tcol).reindex(grid), t, req, g)
            prepared.append(ps)
    if len({f for f in freqs.values() if f}) > 1:
        issues.append(Issue("MIXED_FREQUENCIES", Severity.ERROR, f"series have different frequencies: {freqs}", "Forecast each frequency in a separate request."))
        for p in prepared:
            p.ok = False
    prof["frequencies"] = freqs
    return prof, issues, prepared


def _leakage_screen(frame: pd.DataFrame, target: str, req: ForecastRequest, g) -> list[Issue]:
    out = []
    y = frame[target].astype("float64")
    yf = y.shift(-1)
    for c in frame.columns:
        if c in (target, req.group_column) or c in req.target_columns:
            continue
        if not pd.api.types.is_numeric_dtype(frame[c]):
            continue
        x = frame[c].astype("float64")
        # a column that reproduces a FUTURE target value: compare changes (levels of a trending series correlate with
        # everything), i.e. corr(dx_t, dy_{t+1}) against corr(dx_t, dy_t)
        dx, dy, dyf = x.diff(), y.diff(), yf.diff()
        ok = dx.notna() & dy.notna() & dyf.notna()
        if ok.sum() < 20 or dx[ok].std() == 0 or dy[ok].std() == 0 or dyf[ok].std() == 0:
            if SUSPICIOUS_NAME.search(str(c)):
                out.append(Issue("SUSPICIOUS_COLUMN_NAME", Severity.WARNING, f"column '{c}' has a name suggesting future or label information", "Check how it was computed.", f"column:{c}"))
            continue
        cf, cn = abs(np.corrcoef(dx[ok], dyf[ok])[0, 1]), abs(np.corrcoef(dx[ok], dy[ok])[0, 1])
        if cf >= 0.99 and cf > cn + 0.01:
            out.append(Issue("POSSIBLE_FUTURE_DERIVED_COLUMN", Severity.WARNING, f"changes in column '{c}' match the NEXT change of '{target}' (|corr| {cf:.3f})",
                             "It may be derived from future data; it is not used by the engine, and must not be used as a feature.", f"column:{c}", {"corr_next": cf, "corr_same": cn}))
        elif SUSPICIOUS_NAME.search(str(c)):
            out.append(Issue("SUSPICIOUS_COLUMN_NAME", Severity.WARNING, f"column '{c}' has a name suggesting future or label information", "Check how it was computed.", f"column:{c}"))
    return out
