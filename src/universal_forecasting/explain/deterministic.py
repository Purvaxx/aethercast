"""Deterministic (template) explanation built only from the evidence object. Needs no API key and never invents a
value: a missing field is reported as unavailable. Numbers are rendered from engine values (6 significant digits for
display; the unrounded values stay in `ExplanationResponse.forecast` and in the engine response)."""
from __future__ import annotations

import re

from .evidence import Evidence

NOT_GUARANTEE = "Validation results describe past data from this request and do not guarantee future accuracy."


def fmt(v) -> str:
    if v is None:
        return "unavailable"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return f"{v:.6g}"


def ts(v) -> str:
    return str(v).replace("T", " ") if v else "unavailable"


def sections(ev: Evidence, detailed: bool) -> tuple[dict, list]:
    f, d, b = ev.facts, ev.facts["derived"], f"series[{ev.series_index}]"
    refs = []
    name = f"'{f['target']}'" + (f" (series '{f['group']}')" if f["group"] else "")       # user-supplied labels, quoted as data
    fc = f["forecast"]
    # ---- summary
    if not fc:
        probs = "; ".join(f"{i['code']}: {i['message']}" for i in f["issues"] if i["severity"] == "error") or "no reason was reported"
        return ({"summary": f"The engine did not produce a forecast for {name}. Blocking issues: {probs}", "trend_interpretation": "Unavailable: there is no forecast.",
                 "model_selection_explanation": "Unavailable: no model was selected.", "uncertainty_explanation": "Unavailable: there is no forecast."},
                [{"section": "summary", "paths": [f"{b}.status", f"{b}.issues", "issues"]}])
    summary = (f"The engine forecast {len(fc)} step(s) of {name} at frequency {f['frequency']} ({f['frequency_source']}), from the forecast origin {ts(f['forecast_origin'])}. "
               f"The first forecast, for {ts(fc[0]['timestamp'])}, is {fmt(fc[0]['point'])}; the last, for {ts(fc[-1]['timestamp'])}, is {fmt(fc[-1]['point'])}. "
               f"The selected model is {f['selected_model']}. These are model forecasts, not observed values.")
    refs.append({"section": "summary", "paths": [f"{b}.forecast[0].point", f"{b}.forecast[0].timestamp", f"{b}.forecast[{len(fc) - 1}].point",
                                                 f"{b}.forecast[{len(fc) - 1}].timestamp", f"{b}.forecast_origin", f"{b}.selected_model", f"{b}.frequency"]})
    # ---- trend
    if len(fc) >= 2:
        word = {"up": "rises", "down": "falls", "flat": "stays flat"}[d["forecast_direction"]]
        trend = (f"Across the forecast horizon the point forecast {word}: it changes by {fmt(d['forecast_change_first_to_last'])} from step 1 to step {len(fc)} "
                 f"(lowest {fmt(d['forecast_min'])}, highest {fmt(d['forecast_max'])}). This describes the forecast path only; it is not a statement about causes.")
        refs.append({"section": "trend_interpretation", "paths": ["derived.forecast_change_first_to_last", "derived.forecast_min", "derived.forecast_max"]})
        if f["selected_model"] == "naive":
            trend += " The naive model repeats the last observed value, so a flat path reflects the model, not evidence that the series will not move."
    else:
        trend = "A single forecast step was requested, so there is no path to describe."
    # ---- model selection
    ev_c = [c for c in f["candidates"] if c["status"] == "evaluated"]
    reg = f["registry"].get(f["selected_model"], {})
    ms = f"The engine selected {f['selected_model']}. Engine reason: {f['selection_reason']}."
    paths = [f"{b}.selected_model"]
    if ev_c:
        v = f["validation"]
        parts = []
        for k, c in enumerate(f["candidates"]):
            if c["status"] == "evaluated":
                parts.append(f"{c['model']} MAE {fmt(c['val_mae'])} (the baseline)" if c["model"] == "naive"
                             else f"{c['model']} MAE {fmt(c['val_mae'])} ({fmt(c['rel_mae_vs_naive'])} times the naive MAE)")
                paths += [f"{b}.candidates[{k}].val_mae", f"{b}.candidates[{k}].rel_mae_vs_naive"]
        ms += (f" Candidates were compared on {fmt(v['n_common_origins'])} identical validation origins between {ts(v['first_origin'])} and {ts(v['last_origin'])}: "
               + "; ".join(parts) + ". Lower MAE is better.")
        paths += [f"{b}.validation.n_common_origins", "policy.tie_tolerance"]
        if "selected_vs_lowest_mae_gap_pct" in d:
            if d["lowest_validation_mae_model"] == f["selected_model"]:
                ms += f" {f['selected_model']} had the lowest validation MAE."
            else:
                ms += (f" {d['lowest_validation_mae_model']} had the lowest validation MAE, but {f['selected_model']} was within the engine's tie tolerance of "
                       f"{fmt(d['tie_tolerance_pct'])}% (gap {fmt(d['selected_vs_lowest_mae_gap_pct'])}%), and ties go to the simpler model.")
            paths += ["derived.selected_vs_lowest_mae_gap_pct", "derived.tie_tolerance_pct"]
        if f["selected_model"] == "naive" and len(ev_c) > 1:
            ms += f" No candidate beat the naive baseline by more than the {fmt(d['tie_tolerance_pct'])}% tie tolerance."
    else:
        ms += " No validation comparison was possible, so the naive fallback was used without metrics."
    others = [c for c in f["candidates"] if c["status"] not in ("evaluated", "fallback") and (detailed or c["status"] == "failed")]
    if others:
        ms += " Not evaluated: " + "; ".join(f"{c['model']} ({c['status']}: {c['reason']})" for c in others) + "."
    if reg:
        ms += (f" Evidence level of {f['selected_model']} in the registry: {reg['evidence_level']} ({reg['evidence']}). "
               "This is not a claim that the model is best in general; it was chosen for this series on this validation period.")
    ms += " " + NOT_GUARANTEE
    refs.append({"section": "model_selection_explanation", "paths": paths})
    # ---- uncertainty
    u = f["uncertainty"]
    if u["status"] == "available":
        lv = [str(x) for x in u["levels"]]
        ex = fc[0].get("intervals", {})
        unc = (f"Prediction intervals are available at levels {', '.join(fmt(d['interval_level_pct'][x]) + '%' for x in lv)}. They are {u['method']} "
               f"({fmt(u['n_residuals_per_step'])} residuals per step). For step 1 the intervals are "
               + "; ".join(f"{fmt(d['interval_level_pct'][x])}%: {fmt(ex[x][0])} to {fmt(ex[x][1])}" for x in lv if x in ex)
               + ". Coverage is not guaranteed, the intervals assume future errors resemble validation errors, and a level is not the probability that the forecast is correct.")
        refs.append({"section": "uncertainty_explanation", "paths": [f"{b}.uncertainty.n_residuals_per_step"] +
                     [f"{b}.forecast[0].intervals[{x}][{j}]" for x in lv if x in ex for j in (0, 1)]})
    else:
        unc = f"Prediction intervals are unavailable: {u['reason'] or 'the engine did not report a reason'}. No replacement bounds are provided."
        refs.append({"section": "uncertainty_explanation", "paths": [f"{b}.uncertainty"]})
    return {"summary": summary, "trend_interpretation": trend, "model_selection_explanation": ms, "uncertainty_explanation": unc}, refs


# ------------------------------------------------------------------ questions
NEW_REQUEST = re.compile(r"(what if|instead of|re-?run|different (horizon|model|target|frequency)|another (model|target|horizon)|longer horizon|shorter horizon|"
                         r"\b(arima|sarima|prophet|lstm|gru|xgboost|lightgbm|transformer|neural|mlp)\b|"
                         r"(next|for|over) (the next )?\d+\s*(more )?(steps?|hours?|days?|weeks?|months?|years?|bars?)|\d+\s*[- ]?(step|hour|day|week|month|year)s?[- ]ahead|"
                         r"forecast (for|of) (a |the )?(different|other|another)|can you (forecast|predict)|predict (the )?(price|value) (on|at|for)|add (a |an )?(feature|covariate|variable))", re.I)
INTENTS = [("baseline", re.compile(r"baseline|naive|compare|comparison|better than|versus|\bvs\b", re.I)),
           ("model_selection", re.compile(r"(why|how).*(select|chose|chosen|pick)|why.*model|which model|model (was )?(selected|chosen|used)", re.I)),
           ("intervals", re.compile(r"interval|uncertain|range|bound|band|margin", re.I)),
           ("horizon_difficulty", re.compile(r"(which|what).*(horizon|step).*(hard|difficult|worse|error)|hard(er|est)|difficult|more error", re.I)),
           ("reliability", re.compile(r"reliab|trust|accura|confiden|how good|how sure|dependab", re.I)),
           ("limitations", re.compile(r"limit|caveat|assum|weakness|risk|careful|consider", re.I)),
           ("trend", re.compile(r"trend|direction|going (up|down)|increas|decreas|rise|fall", re.I)),
           ("meaning", re.compile(r"what does|mean|interpret|explain|summar|tell me about|understand", re.I))]


def classify(question: str) -> str:
    if NEW_REQUEST.search(question):
        return "new_request"
    for name, rx in INTENTS:
        if rx.search(question):
            return name
    return "unsupported"


def answer(intent: str, ev: Evidence, secs: dict) -> tuple[bool, str, list, list]:
    """(answerable, answer, missing_evidence, evidence paths) for a supported intent, from evidence only."""
    f, d, b = ev.facts, ev.facts["derived"], f"series[{ev.series_index}]"
    if not f["forecast"]:
        return False, secs["summary"], ["a forecast (the engine returned none for this series)"], [f"{b}.issues"]
    sel = next((c for c in f["candidates"] if c["model"] == f["selected_model"] and c["status"] == "evaluated"), None)
    if intent == "meaning":
        return True, secs["summary"] + " " + secs["trend_interpretation"], [], [f"{b}.forecast"]
    if intent == "trend":
        return True, secs["trend_interpretation"], [], ["derived.forecast_change_first_to_last"]
    if intent == "model_selection":
        return True, secs["model_selection_explanation"], [], [f"{b}.candidates", f"{b}.selection_reason"]
    if intent == "intervals":
        return True, secs["uncertainty_explanation"], ([] if f["uncertainty"]["status"] == "available" else ["prediction intervals"]), [f"{b}.uncertainty"]
    if intent == "limitations":
        lim = " ".join(f"- {x}" for x in f["limitations"]) or "The engine reported no limitations."
        warn = " ".join(f"- {i['code']}: {i['message']}" for i in f["issues"]) or "none"
        return True, f"Limitations reported by the engine: {lim} Warnings: {warn}", [], [f"{b}.limitations", f"{b}.issues"]
    if intent == "baseline":
        if not sel:
            return False, "The engine has no validation comparison for this series, so the selected model cannot be compared with the baseline.", ["validation metrics"], [f"{b}.candidates"]
        if f["selected_model"] == "naive":
            return True, ("The selected model is the naive baseline itself. " + secs["model_selection_explanation"]), [], [f"{b}.candidates"]
        pct = d["mae_change_vs_naive_pct"][f["selected_model"]]
        return True, (f"On the validation period {f['selected_model']} had MAE {fmt(sel['val_mae'])} against {fmt(f['validation']['naive_mae'])} for the naive baseline, "
                      f"which is {fmt(sel['rel_mae_vs_naive'])} times the naive MAE ({fmt(abs(pct))}% {'lower' if pct < 0 else 'higher'}), on "
                      f"{fmt(f['validation']['n_common_origins'])} identical origins. " + NOT_GUARANTEE), [], [f"{b}.validation.naive_mae", f"derived.mae_change_vs_naive_pct[{f['selected_model']}]"]
    if intent == "horizon_difficulty":
        if not sel or not sel.get("per_step_mae") or len(sel["per_step_mae"]) < 2:
            return False, "The engine response has no per-step validation errors to compare horizons.", ["per-step validation errors for the selected model"], []
        ps = sel["per_step_mae"]
        return True, (f"On the validation period the selected model's error was largest at step {fmt(d['hardest_step'])} (MAE {fmt(ps[d['hardest_step'] - 1])}) and smallest at step "
                      f"{fmt(d['easiest_step'])} (MAE {fmt(ps[d['easiest_step'] - 1])}). This is a description of validation errors, not a tested difference."), [], ["derived.hardest_step", "derived.easiest_step"]
    if intent == "reliability":
        if not sel:
            return False, ("The engine could not validate any model for this series, so there is no measured accuracy. The forecast is a naive fallback. "
                           + secs["uncertainty_explanation"]), ["validation metrics"], [f"{b}.validation"]
        warn = [i for i in f["issues"] if i["code"] in ("UNSTABLE_VALIDATION", "DISTRIBUTION_SHIFT", "MISSING_POINTS", "FALLBACK_USED")]
        txt = (f"The engine does not produce a reliability score. What it measured: on {fmt(f['validation']['n_common_origins'])} validation origins the selected model's MAE was "
               f"{fmt(sel['val_mae'])}, {fmt(sel['rel_mae_vs_naive'])} times the naive MAE. " + secs["uncertainty_explanation"])
        if warn:
            txt += " Warnings that reduce confidence in these figures: " + "; ".join(f"{i['code']}: {i['message']}" for i in warn) + "."
        return True, txt + " " + NOT_GUARANTEE, [], [f"{b}.validation", f"{b}.uncertainty", f"{b}.issues"]
    return False, "", [], []
