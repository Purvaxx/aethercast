"""Engine entry point: forecast(request) -> ForecastResponse. Fully numerical; no language model is involved."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from .profiling import PreparedSeries, profile
from .registry import REGISTRY, snapshot
from .schemas import ForecastPoint, ForecastRequest, ForecastResponse, Issue, RequestError, SeriesResult, Severity, Status
from .selection import evaluate, intervals

SCOPE_LIMITATIONS = ["univariate forecasts per (group, target): other columns and known-future covariates are not used in Phase 4A",
                     "horizon counts steps of one regular frequency; business-day / session calendars and holidays are not modelled",
                     "the selected model is chosen on one chronological validation period; validation accuracy is not a guarantee of future accuracy"]


def _series(ps: PreparedSeries, req: ForecastRequest) -> SeriesResult:
    res = SeriesResult(ps.group, ps.target, "error", ps.frequency or None, ps.frequency_source, ps.seasonal_period, issues=list(ps.issues))
    if not ps.ok:
        return res
    y, ts, H = ps.y, ps.timestamps, req.horizon
    obs = np.isfinite(y)
    res.history = {"start": ts[0].isoformat(), "end": ts[-1].isoformat(), "n_grid_points": int(len(y)), "n_observed": int(obs.sum()),
                   "n_missing": int((~obs).sum()), "inserted_grid_points": ps.inserted_points, "seasonal_period_source": ps.seasonal_source}
    ev = evaluate(y, ts, ps.frequency, ps.seasonal_period, H, req.validation_fraction, req.allowed_models)
    res.issues += ev.issues
    res.candidates = ev.candidates
    if ev.split:
        sp = ev.split
        res.training_boundary = {"selection_train_end": ts[sp["train_end"]].isoformat(), "rule": "all parameters used for model selection were fitted on data up to this time"}
        res.validation = {"first_origin": ts[sp["val_start"]].isoformat(), "last_origin": ts[sp["val_end"]].isoformat(), "n_origins": sp["n_val_origins"],
                          "n_common_origins": sp.get("n_common_origins"), "purge": f"labels reach {H} steps beyond each origin; training labels end at selection_train_end",
                          "naive_mae": ev.naive_mae}
    selected, reason = ev.selected, ev.reason
    n = len(y)
    P = np.array([n - 1])
    path = None
    for attempt in ([selected] if selected == "naive" else [selected, "naive"]):
        try:
            mod = M.build(attempt, ps.frequency)
            ok, why = mod.eligible(y, ps.seasonal_period, H)
            if not ok:
                raise RuntimeError(why)
            mod.fit(y, ts, n - 1, ps.seasonal_period, H)
            f = mod.paths(y, ts, P, H)[0]
            if not np.isfinite(f).all():
                raise RuntimeError("non-finite forecast (missing inputs at the forecast origin)")
            path = f
            if attempt != selected:
                res.issues.append(Issue("FALLBACK_USED", Severity.WARNING, f"{selected} could not produce the final forecast; the naive fallback was used",
                                        "See the candidate list for the failure reason.", f"series:{ps.group}|{ps.target}"))
                reason = f"fallback to naive: {selected} failed at the final fit"
                selected = "naive"
                ev.residuals = None
            break
        except Exception as ex:                                  # noqa: BLE001
            res.issues.append(Issue("MODEL_FAILED", Severity.WARNING if attempt != "naive" else Severity.ERROR,
                                    f"{attempt} failed at the final fit: {type(ex).__name__}: {ex}", "", f"series:{ps.group}|{ps.target}"))
    if path is None:
        return res
    unc, off = intervals(ev.residuals, req.interval_levels)
    step = pd.tseries.frequencies.to_offset(ps.frequency)
    fts = [ts[-1] + (k) * step for k in range(1, H + 1)]
    res.forecast = [ForecastPoint(k, fts[k - 1].isoformat(), float(path[k - 1]),
                                  {lv: [float(path[k - 1] + lo[k - 1]), float(path[k - 1] + hi[k - 1])] for lv, (lo, hi) in off.items()} if off else {})
                    for k in range(1, H + 1)]
    res.uncertainty = unc
    res.selected_model, res.selection_reason = selected, reason
    res.forecast_origin = ts[-1].isoformat()
    res.training_boundary["final_fit_end"] = ts[-1].isoformat()
    res.training_boundary["final_fit_note"] = "after selection the chosen model is re-fitted on the full history up to the forecast origin"
    spec = REGISTRY[selected]
    res.limitations = list(spec.limitations) + SCOPE_LIMITATIONS + [f"evidence level of {selected}: {spec.evidence_level} ({spec.evidence})"]
    res.status = "ok"
    return res


def forecast(req: ForecastRequest) -> ForecastResponse:
    resp = ForecastResponse(Status.ERROR.value, registry=snapshot())
    try:
        resp.request = req.echo()
        req.validate()
    except RequestError as ex:
        resp.issues = ex.issues
        return resp
    prof, issues, prepared = profile(req)
    resp.profile, resp.issues = prof, issues
    if any(i.severity == Severity.ERROR for i in issues) and not any(p.ok for p in prepared):
        resp.series = [SeriesResult(p.group, p.target, "error", issues=list(p.issues)) for p in prepared]
        return resp
    for ps in prepared:
        try:
            resp.series.append(_series(ps, req))
        except Exception as ex:                                  # noqa: BLE001
            resp.series.append(SeriesResult(ps.group, ps.target, "error", issues=list(ps.issues) + [
                Issue("INTERNAL_ERROR", Severity.ERROR, f"{type(ex).__name__}: {ex}", "This is an engine defect; no forecast is returned for this series.",
                      f"series:{ps.group}|{ps.target}")]))
    n_ok = sum(s.status == "ok" for s in resp.series)
    resp.status = (Status.OK if n_ok == len(resp.series) and n_ok else Status.PARTIAL if n_ok else Status.ERROR).value
    return resp
