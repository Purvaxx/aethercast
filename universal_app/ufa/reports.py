"""Insights and downloadable reports, built ONLY from stored run records (see history.py).

Nothing here forecasts. The few derived figures are simple, documented comparisons of numbers that are already in the
record, and each is labelled with what it compares:
    direction      last forecast point  versus  the last observed value        (up / down / flat)
    observed move  last observed value  versus  the first of the stored recent values
    forecast range smallest and largest forecast point; widest interval bounds when the engine returned intervals
When a figure cannot be derived (no stored history, no intervals, no validation) it is reported as unavailable.
"""
from __future__ import annotations

import csv
import io
import math

from . import viewmodels as VM
from .history import slug

FLAT_TOL = 1e-9
OTHER_COLUMNS_NOTE = "Other columns in the file were not used: each forecast is built from the target's own past values only."


def ok_series(record: dict) -> list[dict]:
    return [s for s in (record.get("response") or {}).get("series") or [] if isinstance(s, dict) and s.get("status") == "ok" and s.get("forecast")]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def direction(last_observed, last_forecast) -> dict:
    """Forecast end versus the last observed value. 'unavailable' when either number is missing."""
    if not (_finite(last_observed) and _finite(last_forecast)):
        return {"label": "unavailable", "diff": None, "pct": None}
    diff = last_forecast - last_observed
    label = "flat" if abs(diff) <= FLAT_TOL * max(1.0, abs(last_observed)) else ("up" if diff > 0 else "down")
    return {"label": label, "diff": diff, "pct": (diff / abs(last_observed) * 100.0) if last_observed else None}


def series_facts(record: dict, s: dict) -> dict:
    """Everything the Insights page and the report say about one series, with its source."""
    fc = s["forecast"]
    tail = (record.get("observed_tail") or {}).get(str(s.get("group"))) or []
    last_obs = tail[-1][1] if tail else None
    first_obs = tail[0][1] if tail else None
    pts = [p["point"] for p in fc]
    sel = next((c for c in s.get("candidates") or [] if c.get("model") == s.get("selected_model")), {})
    levels = VM.interval_levels(s)
    env = None
    if levels:
        wide = max(levels, key=float)
        env = {"level": wide, "low": min(p["intervals"][wide][0] for p in fc), "high": max(p["intervals"][wide][1] for p in fc)}
    h = s.get("history") or {}
    return {"group": s.get("group"), "target": s.get("target"), "selected_model": s.get("selected_model"), "selection_reason": s.get("selection_reason"),
            "frequency": s.get("frequency"), "frequency_source": s.get("frequency_source"), "horizon": len(fc), "forecast_origin": s.get("forecast_origin"),
            "history_start": h.get("start"), "history_end": h.get("end"), "n_observed": h.get("n_observed"), "n_missing": h.get("n_missing"),
            "first_forecast": fc[0], "last_forecast": fc[-1], "forecast_min": min(pts), "forecast_max": max(pts),
            "last_observed": last_obs, "last_observed_time": tail[-1][0] if tail else None,
            "direction": direction(last_obs, pts[-1]),
            "observed_move": direction(first_obs, last_obs) | {"points": len(tail), "from_time": tail[0][0] if tail else None},
            "val_mae": sel.get("val_mae"), "rel_mae_vs_naive": sel.get("rel_mae_vs_naive"), "n_validation_origins": (s.get("validation") or {}).get("n_common_origins"),
            "intervals": env, "interval_reason": (s.get("uncertainty") or {}).get("reason"),
            "issues": list(s.get("issues") or []), "limitations": list(s.get("limitations") or []),
            "candidates": [c for c in s.get("candidates") or []]}


def overview(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    codes = {}
    for r in ok:
        for i in (r["response"].get("issues") or []) + [i for s in ok_series(r) for i in (s.get("issues") or [])]:
            codes[i.get("code")] = codes.get(i.get("code"), 0) + 1
    dirs = {"up": 0, "down": 0, "flat": 0, "unavailable": 0}
    models = {}
    n_series = n_no_int = n_naive = 0
    for r in ok:
        for s in ok_series(r):
            f = series_facts(r, s)
            n_series += 1
            dirs[f["direction"]["label"]] += 1
            models[f["selected_model"]] = models.get(f["selected_model"], 0) + 1
            n_no_int += f["intervals"] is None
            n_naive += f["selected_model"] == "naive"
    return {"runs": len(records), "completed": len(ok), "failed": len(records) - len(ok), "datasets": len({r["dataset"]["sha256"] for r in records}),
            "series": n_series, "directions": dirs, "models": models, "issue_codes": codes, "without_intervals": n_no_int, "naive_selected": n_naive,
            "demo_runs": sum(r.get("mode") == "demo" for r in records)}


def comparable_groups(records: list[dict]) -> list[list[dict]]:
    """Runs may be compared only when they forecast the SAME file contents, target and series setting. Returns groups of 2+."""
    groups = {}
    for r in records:
        if r.get("status") == "ok":
            key = (r["dataset"]["sha256"], r["config"]["target_column"], r["config"]["group_column"], r["config"]["timestamp_column"])
            groups.setdefault(key, []).append(r)
    return [sorted(g, key=lambda r: r["created_epoch"]) for g in groups.values() if len(g) > 1]


# ------------------------------------------------------------------ downloads
def file_stem(record: dict) -> str:
    """Unique per run: the run id contains the time to the second and a hash."""
    demo = "DEMO_" if record.get("mode") == "demo" else ""
    return f"{demo}forecast_{slug(record['dataset']['name'].rsplit('.', 1)[0], 32)}_{slug(record['config']['target_column'], 24)}_{record['id']}"


def forecast_csv(record: dict) -> bytes:
    """The engine's forecast points exactly as stored: one row per series and step, with bounds where the engine returned them."""
    series = ok_series(record)
    levels = sorted({lv for s in series for lv in VM.interval_levels(s)}, key=float)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    head = ["series", "target", "step", "timestamp", "forecast"]
    for lv in levels:
        head += [f"lower_{float(lv) * 100:g}", f"upper_{float(lv) * 100:g}"]
    w.writerow(head + ["selected_model", "data_kind", "run_id", "mode"])
    for s in series:
        own = set(VM.interval_levels(s))
        for p in s["forecast"]:
            row = ["" if s.get("group") is None else s.get("group"), s.get("target"), p["step"], p["timestamp"], repr(float(p["point"]))]
            for lv in levels:
                row += [repr(float(p["intervals"][lv][0])), repr(float(p["intervals"][lv][1]))] if lv in own else ["", ""]      # blank = not available
            w.writerow(row + [s.get("selected_model"), "predicted", record["id"], record.get("mode")])
    return out.getvalue().encode("utf-8")


def observed_csv(record: dict) -> bytes:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["series", "target", "timestamp", "observed", "data_kind", "run_id"])
    for s in ok_series(record):
        for t, v in (record.get("observed_tail") or {}).get(str(s.get("group"))) or []:
            w.writerow(["" if s.get("group") is None else s.get("group"), s.get("target"), t, repr(float(v)), "observed", record["id"]])
    return out.getvalue().encode("utf-8")


def _svg(record: dict, s: dict, f: dict) -> str:
    """A plain line chart: observed (gray), forecast (white, or green / red by the stated direction), widest interval (band)."""
    tail = ((record.get("observed_tail") or {}).get(str(s.get("group"))) or [])[-120:]
    fc = s["forecast"]
    ys = [v for _, v in tail] + [p["point"] for p in fc]
    lv = f["intervals"]["level"] if f["intervals"] else None
    if lv:
        ys += [p["intervals"][lv][0] for p in fc] + [p["intervals"][lv][1] for p in fc]
    lo, hi = min(ys), max(ys)
    rng = (hi - lo) or 1.0
    n = len(tail) + len(fc)
    W, H, L, B = 860.0, 300.0, 64.0, 26.0
    X = lambda i: L + (W - L - 12) * (i / max(n - 1, 1))                         # noqa: E731
    Y = lambda v: 10 + (H - B - 20) * (1 - (v - lo) / rng)                       # noqa: E731
    obs = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(tail))
    k0 = len(tail)
    pre = [f"{X(k0 - 1):.1f},{Y(tail[-1][1]):.1f}"] if tail else []
    fcp = " ".join(pre + [f"{X(k0 + j):.1f},{Y(p['point']):.1f}" for j, p in enumerate(fc)])
    band = ""
    if lv:
        up = [f"{X(k0 + j):.1f},{Y(p['intervals'][lv][1]):.1f}" for j, p in enumerate(fc)]
        dn = [f"{X(k0 + j):.1f},{Y(p['intervals'][lv][0]):.1f}" for j, p in enumerate(fc)][::-1]
        band = f"<polygon points='{' '.join(up + dn)}' fill='#ffffff' opacity='.16'/>"
    col = {"up": "#22c55e", "down": "#ef4444"}.get(f["direction"]["label"], "#ffffff")
    grid = "".join(f"<line x1='{L}' x2='{W - 12}' y1='{Y(lo + rng * q):.1f}' y2='{Y(lo + rng * q):.1f}' stroke='#262626'/>"
                   f"<text x='{L - 8}' y='{Y(lo + rng * q) + 4:.1f}' text-anchor='end' fill='#a8a8ad' font-size='11'>{VM.safe(VM.fmt_num(lo + rng * q))}</text>" for q in (0, .25, .5, .75, 1))
    xl = (f"<text x='{L}' y='{H - 6}' fill='#a8a8ad' font-size='11'>{VM.safe(str(tail[0][0])[:16].replace('T', ' ')) if tail else ''}</text>"
          f"<text x='{W - 12}' y='{H - 6}' text-anchor='end' fill='#a8a8ad' font-size='11'>{VM.safe(str(fc[-1]['timestamp'])[:16].replace('T', ' '))}</text>")
    split = f"<line x1='{X(k0 - 0.5):.1f}' x2='{X(k0 - 0.5):.1f}' y1='10' y2='{H - B - 10}' stroke='#6b6b70' stroke-dasharray='4 4'/>" if tail else ""
    return (f"<svg viewBox='0 0 {W:.0f} {H:.0f}' role='img' aria-label='Observed values and forecast' style='width:100%;height:auto;background:#000'>{grid}{band}{split}"
            f"<polyline points='{obs}' fill='none' stroke='#9a9aa0' stroke-width='1.5'/><polyline points='{fcp}' fill='none' stroke='{col}' stroke-width='2.4'/>{xl}</svg>")


def _dir_text(d: dict, what: str) -> str:
    if d["label"] == "unavailable":
        return "unavailable (no stored observed values)"
    amount = VM.fmt_num(d["diff"]) + (f" ({d['pct']:+.2f}%)" if d.get("pct") is not None else "")
    return {"up": f"higher by {amount}", "down": f"lower by {VM.fmt_num(abs(d['diff']))}" + (f" ({d['pct']:+.2f}%)" if d.get("pct") is not None else ""), "flat": "unchanged"}[d["label"]] + f" {what}"


def report_html(record: dict) -> str:
    """A self-contained, human-readable report (no scripts, no external files). All text from the file or the engine is escaped."""
    e, c, ds = VM.safe, record["config"], record["dataset"]
    demo = record.get("mode") == "demo"
    css = ("body{background:#000;color:#f5f5f5;font-family:'Segoe UI',system-ui,-apple-system,Arial,sans-serif;margin:0;padding:32px;line-height:1.5}"
           "main{max-width:920px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}h2{font-size:15px;text-transform:uppercase;letter-spacing:.1em;color:#a8a8ad;margin:30px 0 10px;"
           "border-top:1px solid #2a2a2c;padding-top:18px}p,li{color:#d4d4d8;font-size:14px}small,.m{color:#a8a8ad}table{border-collapse:collapse;width:100%;font-size:13px}"
           "th{text-align:left;color:#a8a8ad;font-weight:600;border-bottom:1px solid #2a2a2c;padding:6px 8px}td{border-bottom:1px solid #1c1c1e;padding:6px 8px;vertical-align:top}"
           "td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.tag{display:inline-block;border:1px solid #6b6b70;border-radius:6px;padding:1px 7px;font-size:11px;"
           "letter-spacing:.06em;text-transform:uppercase;margin-right:6px}.demo{border:2px solid #fff;padding:10px 14px;font-weight:700;margin:0 0 18px}"
           ".obs{color:#9a9aa0}.up{color:#22c55e}.down{color:#ef4444}@media print{body{background:#fff;color:#000}p,li,td{color:#000}svg{background:#000}}")
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>AetherCast AI forecast report {e(record['id'])}</title><style>{css}</style></head><body><main>"]
    if demo:
        out.append("<div class='demo'>DEMO OUTPUT. This is a stored result for the bundled synthetic sample. It is not a forecast of your data.</div>")
    out.append(f"<div class='m' style='letter-spacing:.18em;font-size:12px;font-weight:700'>AetherCast AI</div><h1>Forecast report</h1><p class='m'>Run {e(record['id'])} · generated {e(record['created'])}"
               + (f" · computed in {record['seconds']:.2f} s" if _finite(record.get("seconds")) else "") + "</p>")
    freq = c.get("frequency") or "inferred by the engine from the timestamps"
    out.append("<h2>Request</h2><table>" + "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in (
        ("Dataset", f"{ds['name']} ({ds['rows']:,} rows, {ds['columns']} columns)"), ("File fingerprint (SHA-256)", ds["sha256"]), ("Timestamp column", c["timestamp_column"]),
        ("Target column", c["target_column"]), ("Series identifier", c["group_column"] or "none (one series)"), ("Frequency setting", freq),
        ("Horizon", f"{c['horizon']} steps"), ("Mode", "DEMO (stored result)" if demo else "live forecasting engine"))) + "</table>")
    for i in (record["response"].get("issues") or []):
        out.append(f"<p><span class='tag'>{e(i.get('severity'))}</span>{e(i.get('code'))}: {e(i.get('message'))}</p>")
    for s in ok_series(record):
        f = series_facts(record, s)
        name = f"{f['target']}" + (f" · {f['group']}" if f["group"] is not None else "")
        out.append(f"<h2>Series: {e(name)}</h2>{_svg(record, s, f)}")
        colword = {"up": "green", "down": "red"}.get(f["direction"]["label"], "white")
        out.append(f"<p class='m'>Gray line: observed values from the file (the last {f['observed_move']['points']} stored). {colword.capitalize()} line: the forecast"
                   + (f"; it is {colword} because the last forecast is {'above' if colword == 'green' else 'below'} the last observed value" if colword != "white" else "")
                   + ". " + (f"Band: the {float(f['intervals']['level']):.0%} prediction interval." if f["intervals"] else "No interval band: the engine returned no intervals.")
                   + " The dashed line separates observed from predicted.</p>")
        rows = [("Selected model", f["selected_model"]), ("Why this model", f["selection_reason"]),
                ("Frequency used", f"{f['frequency']} ({f['frequency_source']})"), ("History used by the engine", f"{str(f['history_start'])[:16].replace('T', ' ')} to {str(f['history_end'])[:16].replace('T', ' ')}; "
                 f"{VM.fmt_num(f['n_observed'])} observed, {VM.fmt_num(f['n_missing'])} missing"),
                ("Forecast origin (last observation used)", str(f["forecast_origin"])[:16].replace("T", " ")),
                ("Predicted: forecast range", f"{VM.fmt_num(f['forecast_min'])} to {VM.fmt_num(f['forecast_max'])}"),
                ("Predicted versus observed: last forecast against the last observed value", _dir_text(f["direction"], "than the last observed value") +
                 (f" ({VM.fmt_num(f['last_observed'])})" if f["last_observed"] is not None else "")),
                ("Observed: last value against the first of the stored recent values", _dir_text(f["observed_move"], f"over the last {f['observed_move']['points']} observed points")),
                ("Validation MAE of the selected model", VM.fmt_num(f["val_mae"]) + (f" on {VM.fmt_num(f['n_validation_origins'])} validation origins" if f["n_validation_origins"] else "")),
                ("Validation MAE relative to the naive baseline", (VM.fmt_num(f["rel_mae_vs_naive"]) + " times the naive error (below 1 is better than the baseline)") if _finite(f["rel_mae_vs_naive"]) else "unavailable"),
                ("Prediction intervals", f"available; widest level {float(f['intervals']['level']):.0%}, from {VM.fmt_num(f['intervals']['low'])} to {VM.fmt_num(f['intervals']['high'])} across the horizon"
                 if f["intervals"] else f"unavailable: {f['interval_reason'] or 'the engine gave no reason'}")]
        out.append("<table>" + "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in rows) + "</table>")
        out.append("<p class='m'>Validation figures were measured on a validation period inside this file. They are not test results and do not guarantee future accuracy. "
                   "An interval level is not the probability that the forecast is right.</p>")
        levels = VM.interval_levels(s)
        head = "".join(f"<th class='n'>Lower {float(lv):.0%}</th><th class='n'>Upper {float(lv):.0%}</th>" for lv in levels)
        body = "".join(f"<tr><td class='n'>{p['step']}</td><td>{e(str(p['timestamp']).replace('T', ' '))}</td><td class='n'>{e(VM.fmt_num(p['point']))}</td>"
                       + "".join(f"<td class='n'>{e(VM.fmt_num(p['intervals'][lv][0]))}</td><td class='n'>{e(VM.fmt_num(p['intervals'][lv][1]))}</td>" for lv in levels) + "</tr>" for p in s["forecast"])
        out.append(f"<h2>Predicted values</h2><table><tr><th class='n'>Step</th><th>Time</th><th class='n'>Forecast</th>{head}</tr>{body}</table>"
                   "<p class='m'>Rounded for reading. The exact values are in the CSV download.</p>")
        cand = "".join(f"<tr><td>{e(x.get('model'))}{' (selected)' if x.get('model') == f['selected_model'] else ''}</td><td>{e(x.get('status'))}</td>"
                       f"<td class='n'>{e(VM.fmt_num(x.get('val_mae')))}</td><td class='n'>{e(VM.fmt_num(x.get('val_rmse')))}</td><td class='n'>{e(VM.fmt_num(x.get('rel_mae_vs_naive')))}</td>"
                       f"<td>{e(x.get('evidence_level'))}</td><td>{e(x.get('reason') or '')}</td></tr>" for x in f["candidates"])
        out.append("<h2>Model comparison on the validation period</h2><table><tr><th>Model</th><th>Status</th><th class='n'>Validation MAE</th><th class='n'>Validation RMSE</th>"
                   f"<th class='n'>MAE ÷ naive</th><th>Evidence level</th><th>Note</th></tr>{cand}</table>"
                   "<p class='m'>Every model was scored on the same forecast origins. 'unavailable' means the model was not evaluated on this series.</p>")
        tail = ((record.get("observed_tail") or {}).get(str(s.get("group"))) or [])[-12:]
        if tail:
            out.append("<h2>Observed values (most recent)</h2><table><tr><th>Time</th><th class='n'>Observed</th></tr>"
                       + "".join(f"<tr><td class='obs'>{e(str(t).replace('T', ' '))}</td><td class='n obs'>{e(VM.fmt_num(v))}</td></tr>" for t, v in tail)
                       + f"</table><p class='m'>The last 12 of {f['observed_move']['points']} stored observed values. The full stored set is in the observed-values CSV.</p>")
        out.append("<h2>Warnings</h2>" + ("".join(f"<p><span class='tag'>{e(i.get('severity'))}</span>{e(i.get('code'))}: {e(i.get('message'))}</p>" for i in f["issues"])
                                           or "<p>The engine reported no warnings for this series.</p>"))
        out.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in f["limitations"]) + f"<li>{e(OTHER_COLUMNS_NOTE)}</li></ul>")
    ex = record.get("explanation")
    if ex and ex.get("status") == "ok":
        out.append("<h2>Explanation</h2><p class='m'>Written by fixed rules from the engine's output. No language model was used.</p>"
                   + "".join(f"<p><b>{e(t)}.</b> {e(ex.get(k))}</p>" for t, k in (("Summary", "summary"), ("Trend", "trend_interpretation"), ("Model selection", "model_selection_explanation"),
                                                                                   ("Uncertainty", "uncertainty_explanation"))))
    out.append("<h2>About this report</h2><p class='m'>Every number comes from the forecasting engine's stored response for this run, or is a comparison of two such numbers as labelled. "
               "Forecasts are estimates, not guarantees, and this report is not financial advice.</p></main></body></html>")
    return "".join(out)
