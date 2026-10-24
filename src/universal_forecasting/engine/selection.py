"""Leakage-safe chronological evaluation, deterministic conservative model selection, fallback and validation-residual
prediction intervals for ONE regular-grid series.

Split (n grid points, horizon H, V validation origins):
    validation origins   p in [val_start, n-1-H]           (each has all H labels inside the data)
    training data        y[0 .. val_start]                 (every fitted parameter, scaler and coefficient uses only this)
    purge                a training label lies at t+k <= val_start, so no validation label can enter fitting; a validation
                         forecast from origin p reads y[0..p] only.
Selection (deterministic, training/validation only; no test period exists inside the engine):
    score = validation MAE over all steps on the COMMON origins (every evaluated model and all labels finite);
    the lowest score wins, but any candidate within TIE_TOL (2%) of the best counts as tied and the tie goes to the
    lowest registry cost_rank (simpler / cheaper), so naive wins every effective tie;
    naive is always the fallback when a model fails, is ineligible, or no validation is possible.
After selection the chosen model is re-fitted on the full history to produce the forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import models as M
from .registry import REGISTRY
from .schemas import CandidateResult, Issue, Severity

TIE_TOL = 0.02
MIN_VAL_ORIGINS = 10
MIN_TRAIN_POINTS = 20
MAX_MISSING_FORECAST_SHARE = 0.10
MIN_RESIDUALS_FOR_INTERVAL = 30
MAX_VAL_ORIGINS = 2000


@dataclass
class Evaluation:
    candidates: list
    selected: str
    reason: str
    split: dict
    residuals: np.ndarray | None        # [n_common, H] of the selected model (actual - forecast)
    issues: list = field(default_factory=list)
    naive_mae: float | None = None


def plan_split(n: int, H: int, frac: float) -> dict | None:
    last_origin = n - 1 - H
    V = min(int(n * frac), MAX_VAL_ORIGINS, last_origin - MIN_TRAIN_POINTS - H + 1)
    if V < MIN_VAL_ORIGINS:
        return None
    val_start = last_origin - V + 1
    return {"train_end": val_start, "val_start": val_start, "val_end": last_origin, "n_val_origins": V}


def candidate_names(allowed) -> list[str]:
    names = [m.name for m in sorted(REGISTRY.values(), key=lambda s: s.cost_rank) if m.routed]
    return names


def evaluate(y: np.ndarray, ts, freq: str, m: int | None, H: int, frac: float, allowed: list[str] | None) -> Evaluation:
    n = len(y)
    issues, cands = [], []
    for name, spec in REGISTRY.items():
        if not spec.routed:
            cands.append(CandidateResult(name, "ineligible", "not routed by the engine: " + spec.evidence, evidence_level=spec.evidence_level))
    names = candidate_names(allowed)
    if allowed is not None:
        unknown = [a for a in allowed if a not in REGISTRY]
        if unknown:
            issues.append(Issue("UNKNOWN_MODEL", Severity.WARNING, f"unknown models ignored: {unknown}", f"Known models: {sorted(REGISTRY)}"))
        for nme in names:
            if nme not in allowed and nme != "naive":
                cands.append(CandidateResult(nme, "excluded_by_user", "not in allowed_models", evidence_level=REGISTRY[nme].evidence_level))
        names = [x for x in names if x in allowed or x == "naive"]          # naive always stays as baseline and fallback
    sp = plan_split(n, H, frac)
    if sp is None:
        issues.append(Issue("VALIDATION_NOT_POSSIBLE", Severity.WARNING, f"history too short for a chronological validation with horizon {H} ({n} grid points)",
                            "The naive fallback is returned without validation metrics; provide a longer history for model comparison."))
        for nme in names:
            if nme != "naive":
                cands.append(CandidateResult(nme, "ineligible", "no validation possible: cannot be compared with the baseline", evidence_level=REGISTRY[nme].evidence_level))
        cands.append(CandidateResult("naive", "fallback", "used without validation", evidence_level=REGISTRY["naive"].evidence_level))
        return Evaluation(cands, "naive", "fallback: no validation possible", {}, None, issues)
    te = sp["train_end"]
    P = np.arange(sp["val_start"], sp["val_end"] + 1)
    labels = np.stack([y[P + k] for k in range(1, H + 1)], axis=1)
    preds = {}
    for nme in names:
        mod = M.build(nme, freq)
        ok, why = mod.eligible(y[: te + 1], m, H)
        if not ok:
            cands.append(CandidateResult(nme, "ineligible", why, evidence_level=REGISTRY[nme].evidence_level))
            continue
        try:
            mod.fit(y, ts, te, m, H)
            f = mod.paths(y, ts, P, H)
        except Exception as ex:                                   # noqa: BLE001  (recorded, never substituted)
            cands.append(CandidateResult(nme, "failed", f"{type(ex).__name__}: {ex}", evidence_level=REGISTRY[nme].evidence_level))
            continue
        miss = float((~np.isfinite(f).all(axis=1)).mean())
        if miss > MAX_MISSING_FORECAST_SHARE:
            cands.append(CandidateResult(nme, "failed", f"no forecast for {miss:.0%} of validation origins (missing inputs)", evidence_level=REGISTRY[nme].evidence_level))
            continue
        preds[nme] = f
    if "naive" not in preds:
        issues.append(Issue("BASELINE_UNAVAILABLE", Severity.ERROR, "the naive baseline could not be computed on the validation origins", "Check for long runs of missing values."))
        return Evaluation(cands, "naive", "fallback: baseline unavailable", sp, None, issues)
    common = np.isfinite(labels).all(axis=1)
    for f in preds.values():
        common &= np.isfinite(f).all(axis=1)
    nc = int(common.sum())
    if nc < MIN_VAL_ORIGINS:
        issues.append(Issue("TOO_FEW_COMMON_ORIGINS", Severity.WARNING, f"only {nc} validation origins are usable by every model", "Naive fallback returned."))
        cands += [CandidateResult(k, "ineligible" if k != "naive" else "fallback", "too few common validation origins", evidence_level=REGISTRY[k].evidence_level) for k in preds]
        return Evaluation(cands, "naive", "fallback: too few common validation origins", sp | {"n_common_origins": nc}, None, issues)
    tr = y[: te + 1]
    d = np.abs(np.diff(tr))
    denom = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
    err = {k: (labels - f)[common] for k, f in preds.items()}
    mae = {k: float(np.mean(np.abs(e))) for k, e in err.items()}
    for k, e in err.items():
        cands.append(CandidateResult(k, "evaluated", "", mae[k], float(np.sqrt(np.mean(e ** 2))), mae[k] / mae["naive"] if mae["naive"] > 0 else None,
                                     mae[k] / denom if denom and denom > 0 else None, np.mean(np.abs(e), axis=0).round(10).tolist(), nc, REGISTRY[k].evidence_level))
    best = min(mae.values())
    tied = [k for k, v in mae.items() if v <= best * (1 + TIE_TOL) + 1e-15]
    sel = min(tied, key=lambda k: (REGISTRY[k].cost_rank, k))
    lowest = min(mae, key=lambda k: (mae[k], REGISTRY[k].cost_rank))
    if sel == lowest:
        reason = f"lowest validation MAE ({mae[sel]:.6g}) on {nc} common origins"
    else:
        reason = (f"{lowest} had the lowest validation MAE ({mae[lowest]:.6g}) but {sel} is within {TIE_TOL:.0%} ({mae[sel]:.6g}); "
                  f"ties go to the simpler model")
    # stability: does the selected model beat naive in both halves of the validation period?
    if sel != "naive":
        half = nc // 2
        a = np.mean(np.abs(err[sel][:half])) / max(np.mean(np.abs(err["naive"][:half])), 1e-300)
        b = np.mean(np.abs(err[sel][half:])) / max(np.mean(np.abs(err["naive"][half:])), 1e-300)
        if (a < 1) != (b < 1):
            issues.append(Issue("UNSTABLE_VALIDATION", Severity.WARNING, f"{sel} beats naive in only one half of the validation period (relative MAE {a:.3f} vs {b:.3f})",
                                "Treat the model choice as fragile.", details={"first_half": float(a), "second_half": float(b)}))
    # distribution shift: validation-period level/scale vs training
    v = y[sp["val_start"] + 1:]
    if np.isfinite(tr).sum() > 5 and np.isfinite(v).sum() > 5 and np.nanstd(tr) > 0:
        z = abs(np.nanmean(v) - np.nanmean(tr)) / np.nanstd(tr)
        ratio = np.nanstd(v) / np.nanstd(tr)
        if z > 2 or ratio > 2 or ratio < 0.5:
            issues.append(Issue("DISTRIBUTION_SHIFT", Severity.WARNING, f"the validation period differs from the training period (mean shift {z:.1f} sd, sd ratio {ratio:.2f})",
                                "Validation metrics may not describe future accuracy.", details={"mean_shift_sd": float(z), "sd_ratio": float(ratio)}))
    sp = sp | {"n_common_origins": nc}
    return Evaluation(cands, sel, reason, sp, err[sel], issues, mae["naive"])


def intervals(residuals: np.ndarray | None, levels) -> tuple[dict, dict | None]:
    """Empirical quantiles of the selected model's validation residuals per step. Returns (uncertainty block, offsets)."""
    if residuals is None:
        return {"status": "unavailable", "reason": "no validation residuals (fallback forecast without validation)"}, None
    n = residuals.shape[0]
    if n < MIN_RESIDUALS_FOR_INTERVAL:
        return {"status": "unavailable", "reason": f"only {n} validation residuals per step (minimum {MIN_RESIDUALS_FOR_INTERVAL})", "n_residuals_per_step": n}, None
    off = {str(lv): (np.quantile(residuals, (1 - lv) / 2, axis=0), np.quantile(residuals, 1 - (1 - lv) / 2, axis=0)) for lv in levels}
    return {"status": "available", "method": "empirical quantiles of validation residuals of the selected model, per forecast step",
            "n_residuals_per_step": n, "levels": [float(lv) for lv in levels],
            "caveats": ["coverage is NOT guaranteed: residuals come from one validation period with overlapping forecast origins",
                        "intervals assume future errors resemble validation errors (no distribution shift)",
                        "these are not calibrated probabilities"],
            "validation_residual_summary": {"mean": float(np.mean(residuals)), "sd": float(np.std(residuals)), "mae_by_step": np.mean(np.abs(residuals), axis=0).tolist()}}, off
