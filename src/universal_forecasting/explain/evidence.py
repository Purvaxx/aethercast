"""Evidence object: the minimal, traceable subset of a Phase 4A response needed to explain ONE series.

`facts`   nested dict sent to the language model (aggregated engine outputs only: no raw dataset rows exist in an
          engine response, and nothing beyond the selected series is included).
`index`   {path: value} for every fact, where `path` points into the engine response (e.g. "series[0].forecast[2].point").
          Derived values (documented formulas over engine fields, computed here by code, never by the model) use
          paths starting with "derived.".
Untrusted strings (target / group / column names) are length-capped and stripped of control characters; they are
data, never instructions.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..engine.selection import MIN_RESIDUALS_FOR_INTERVAL, TIE_TOL

_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean_text(s, cap=120) -> str:
    return _CTRL.sub(" ", str(s))[:cap]


@dataclass
class Evidence:
    series_index: int
    facts: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    unavailable: list = field(default_factory=list)

    def add(self, path: str, value):
        if value is not None and not (isinstance(value, float) and not math.isfinite(value)):
            self.index[path] = value
        return value

    def numbers(self) -> dict:
        return {p: float(v) for p, v in self.index.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def timestamps(self) -> list[str]:
        return [v for p, v in self.index.items() if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v)]


def select_series(resp: dict, group, target) -> int:
    ser = resp["series"]
    if not ser:
        return -1
    for i, s in enumerate(ser):
        if (group is None or s.get("group") == group) and (target is None or s.get("target") == target):
            return i
    return -2


def build(resp: dict, i: int) -> Evidence:
    s = resp["series"][i]
    ev = Evidence(i)
    b = f"series[{i}]"
    a = ev.add
    facts = {"target": clean_text(s.get("target")), "group": clean_text(s["group"]) if s.get("group") is not None else None,
             "series_status": a(f"{b}.status", s.get("status")), "frequency": a(f"{b}.frequency", s.get("frequency")),
             "frequency_source": s.get("frequency_source"), "seasonal_period": a(f"{b}.seasonal_period", s.get("seasonal_period")),
             "horizon_steps": a("request.horizon", resp.get("request", {}).get("horizon")), "horizon_unit": resp.get("request", {}).get("horizon_unit"),
             "forecast_origin": a(f"{b}.forecast_origin", s.get("forecast_origin")),
             "selected_model": a(f"{b}.selected_model", s.get("selected_model")), "selection_reason": s.get("selection_reason"),
             "baseline_model": s.get("baseline_model")}
    h = s.get("history") or {}
    facts["history"] = {k: a(f"{b}.history.{k}", h.get(k)) for k in ("start", "end", "n_grid_points", "n_observed", "n_missing", "inserted_grid_points")}
    v = s.get("validation") or {}
    facts["validation"] = {k: a(f"{b}.validation.{k}", v.get(k)) for k in ("first_origin", "last_origin", "n_origins", "n_common_origins", "naive_mae")} if v else None
    if not v:
        ev.unavailable.append("validation metrics (no chronological validation was possible)")
    tb = s.get("training_boundary") or {}
    facts["training_boundary"] = {k: a(f"{b}.training_boundary.{k}", tb.get(k)) for k in ("selection_train_end", "final_fit_end")}
    # forecast path
    fc = s.get("forecast") or []
    facts["forecast"] = []
    for k, p in enumerate(fc):
        item = {"step": a(f"{b}.forecast[{k}].step", p["step"]), "timestamp": a(f"{b}.forecast[{k}].timestamp", p["timestamp"]), "point": a(f"{b}.forecast[{k}].point", p["point"])}
        for lv, (lo, hi) in (p.get("intervals") or {}).items():
            item.setdefault("intervals", {})[lv] = [a(f"{b}.forecast[{k}].intervals[{lv}][0]", lo), a(f"{b}.forecast[{k}].intervals[{lv}][1]", hi)]
        facts["forecast"].append(item)
    if not fc:
        ev.unavailable.append("forecast values (the engine produced no forecast for this series)")
    # candidates
    facts["candidates"] = []
    for k, c in enumerate(s.get("candidates") or []):
        cb = f"{b}.candidates[{k}]"
        item = {"model": c["model"], "status": c["status"], "reason": clean_text(c.get("reason", ""), 300), "evidence_level": c.get("evidence_level"),
                "val_mae": a(f"{cb}.val_mae", c.get("val_mae")), "val_rmse": a(f"{cb}.val_rmse", c.get("val_rmse")),
                "rel_mae_vs_naive": a(f"{cb}.rel_mae_vs_naive", c.get("rel_mae_vs_naive")), "mase": a(f"{cb}.mase", c.get("mase")),
                "n_origins": a(f"{cb}.n_origins", c.get("n_origins")) if c.get("n_origins") else None}
        if c.get("per_step_mae"):
            item["per_step_mae"] = [a(f"{cb}.per_step_mae[{j}]", x) for j, x in enumerate(c["per_step_mae"])]
        facts["candidates"].append(item)
    # uncertainty
    u = s.get("uncertainty") or {}
    facts["uncertainty"] = {"status": u.get("status"), "method": u.get("method"), "reason": u.get("reason"), "caveats": u.get("caveats"),
                            "n_residuals_per_step": a(f"{b}.uncertainty.n_residuals_per_step", u.get("n_residuals_per_step")),
                            "levels": [a(f"{b}.uncertainty.levels[{j}]", x) for j, x in enumerate(u.get("levels") or [])]}
    if u.get("status") != "available":
        ev.unavailable.append("prediction intervals (" + (u.get("reason") or "not reported by the engine") + ")")
    facts["issues"] = [{"code": x["code"], "severity": x["severity"], "message": clean_text(x["message"], 300)} for x in (resp.get("issues") or []) + (s.get("issues") or [])]
    facts["limitations"] = list(s.get("limitations") or [])
    facts["policy"] = {"tie_tolerance": a("policy.tie_tolerance", TIE_TOL), "min_residuals_for_intervals": a("policy.min_residuals_for_intervals", MIN_RESIDUALS_FOR_INTERVAL),
                       "tie_rule": "candidates within the tie tolerance of the lowest validation MAE are tied; ties go to the simpler model, so naive wins effective ties",
                       "metrics_are": "chronological VALIDATION metrics on this request's own data (not test metrics)"}
    reg = {m["name"]: m for m in resp.get("registry") or []}
    facts["registry"] = {c["model"]: {"evidence_level": reg[c["model"]]["evidence_level"], "routed": reg[c["model"]]["routed"], "evidence": reg[c["model"]]["evidence"]}
                         for c in facts["candidates"] if c["model"] in reg}
    facts["derived"] = _derived(ev, facts)
    ev.facts = facts
    return ev


def _derived(ev: Evidence, f: dict) -> dict:
    """Deterministic derivations over engine fields (code, not the language model). Each has a documented formula."""
    d, a = {}, ev.add
    fc = f["forecast"]
    if len(fc) >= 2:
        first, last = fc[0]["point"], fc[-1]["point"]
        d["forecast_change_first_to_last"] = a("derived.forecast_change_first_to_last", last - first)          # point[H] - point[1]
        d["forecast_direction"] = "flat" if last == first else ("up" if last > first else "down")
        d["forecast_min"], d["forecast_max"] = a("derived.forecast_min", min(p["point"] for p in fc)), a("derived.forecast_max", max(p["point"] for p in fc))
    for p in fc:
        for lv, (lo, hi) in (p.get("intervals") or {}).items():
            d.setdefault("interval_width", {}).setdefault(lv, []).append(a(f"derived.interval_width[{lv}][step{p['step']}]", hi - lo))   # hi - lo
    sel = next((c for c in f["candidates"] if c["model"] == f["selected_model"] and c["status"] == "evaluated"), None)
    ev_c = [c for c in f["candidates"] if c["status"] == "evaluated"]
    if ev_c:
        best = min(ev_c, key=lambda c: c["val_mae"])
        d["lowest_validation_mae_model"] = best["model"]
        for c in ev_c:
            if c.get("rel_mae_vs_naive") is not None:
                d.setdefault("mae_change_vs_naive_pct", {})[c["model"]] = a(f"derived.mae_change_vs_naive_pct[{c['model']}]", (c["rel_mae_vs_naive"] - 1) * 100)   # (rel - 1) * 100
        if sel and best["val_mae"] and best["val_mae"] > 0:
            gap = sel["val_mae"] / best["val_mae"] - 1
            d["selected_vs_lowest_mae_gap_pct"] = a("derived.selected_vs_lowest_mae_gap_pct", gap * 100)                 # (selected / lowest - 1) * 100
            d["selected_within_tie_tolerance_of_lowest"] = bool(gap <= f["policy"]["tie_tolerance"])
        d["tie_tolerance_pct"] = a("derived.tie_tolerance_pct", f["policy"]["tie_tolerance"] * 100)
    if sel and sel.get("per_step_mae"):
        ps = sel["per_step_mae"]
        j = max(range(len(ps)), key=lambda k: ps[k])
        e = min(range(len(ps)), key=lambda k: ps[k])
        d["hardest_step"], d["easiest_step"] = a("derived.hardest_step", j + 1), a("derived.easiest_step", e + 1)      # argmax / argmin of per_step_mae
    else:
        ev.unavailable.append("per-step validation errors for the selected model")
    for lv in f["uncertainty"]["levels"]:
        a(f"derived.interval_level_pct[{lv}]", lv * 100)
        d.setdefault("interval_level_pct", {})[str(lv)] = lv * 100
    return d
