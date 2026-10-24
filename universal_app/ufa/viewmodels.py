"""View models: turn a Phase 4A / 4B response into safe, display-ready pieces. Pure functions, no Streamlit.

The interface never computes or alters a forecast number. These functions only (a) check that a response is
structurally sound before it is displayed, (b) lay values out for charts and tables, and (c) escape text.
A response that fails the integrity check is not charted: the reasons are shown instead.
"""
from __future__ import annotations

import html
import math
import re

import pandas as pd

MAX_HISTORY_POINTS = 2000
MD_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-!|>~<&])")
VALIDATION_LABELS = {"val_mae": "Validation MAE", "val_rmse": "Validation RMSE", "rel_mae_vs_naive": "Validation MAE ÷ naive MAE", "mase": "Validation MASE",
                     "n_origins": "Validation origins"}


def safe(text, cap: int = 2000) -> str:
    """Escape HTML so untrusted text (column names, engine messages, model wording) is shown literally."""
    return html.escape("" if text is None else str(text)[:cap], quote=True)


def safe_md(text, cap: int = 4000) -> str:
    """Escape HTML and Markdown control characters: no links, images, headings or HTML can be injected."""
    return MD_SPECIALS.sub(r"\\\1", html.escape("" if text is None else str(text)[:cap], quote=False))


def fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "unavailable"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.6g}"


def check_forecast_response(resp) -> list[str]:
    """Structural integrity of a Phase 4A response. Returns problems (empty = displayable)."""
    p = []
    if not isinstance(resp, dict):
        return ["the response is not an object"]
    if resp.get("schema_version") != "forecast-response-v1":
        p.append(f"unexpected schema_version {resp.get('schema_version')!r}")
    if resp.get("status") not in ("ok", "partial", "error"):
        p.append("missing or unknown status")
    if not isinstance(resp.get("series"), list) or not isinstance(resp.get("issues"), list):
        p.append("series / issues are missing")
        return p
    for i, s in enumerate(resp["series"]):
        fc = s.get("forecast") or []
        last = None
        for k, pt in enumerate(fc):
            v = pt.get("point")
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                p.append(f"series {i} step {k + 1}: forecast value is not a finite number")
            t = pd.to_datetime(pt.get("timestamp"), errors="coerce")
            if pd.isna(t):
                p.append(f"series {i} step {k + 1}: timestamp is not a date")
            elif last is not None and t <= last:
                p.append(f"series {i} step {k + 1}: timestamps are not increasing")
            last = t if not pd.isna(t) else last
            if pt.get("step") != k + 1:
                p.append(f"series {i}: step numbers are not 1..H")
            for lv, b in (pt.get("intervals") or {}).items():
                ok = isinstance(b, (list, tuple)) and len(b) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in b)
                if not ok or b[0] > b[1]:
                    p.append(f"series {i} step {k + 1}: interval {lv} is malformed")
                elif isinstance(v, (int, float)) and math.isfinite(v) and not (b[0] <= v <= b[1]):
                    p.append(f"series {i} step {k + 1}: point forecast lies outside its {lv} interval")
        if s.get("status") == "ok" and not fc:
            p.append(f"series {i}: marked ok but has no forecast")
        if (s.get("uncertainty") or {}).get("status") != "available" and any(pt.get("intervals") for pt in fc):
            p.append(f"series {i}: intervals present although uncertainty is not 'available'")
        if s.get("selected_model") and s.get("candidates") and s["selected_model"] not in [c.get("model") for c in s["candidates"]]:
            p.append(f"series {i}: selected model is not among the candidates")
    return list(dict.fromkeys(p))[:20]


def check_explanation(expl, series: dict) -> list[str]:
    """The explanation must carry the engine's forecast points unchanged; otherwise it is not shown."""
    p = []
    if not isinstance(expl, dict) or expl.get("schema_version") != "explanation-response-v1":
        return ["the explanation response has an unexpected format"]
    if expl.get("status") == "ok" and series.get("forecast") and expl.get("forecast") != series.get("forecast"):
        p.append("the explanation's forecast values differ from the engine's forecast")
    if expl.get("source") not in ("deterministic_template", "deterministic_fallback", "llm_validated"):
        p.append("unknown explanation source")
    return p


def history_frame(frame: pd.DataFrame, ts_col: str, target: str, group_col=None, group=None) -> pd.DataFrame:
    """Observed values exactly as uploaded (no filling); thinned for display only when very long (every k-th point, stated in the UI)."""
    d = frame if not group_col else frame[frame[group_col].astype(str) == str(group)]
    out = pd.DataFrame({"timestamp": pd.to_datetime(d[ts_col], errors="coerce"), "value": pd.to_numeric(d[target], errors="coerce")}).dropna(subset=["timestamp"]).sort_values("timestamp")
    out["kind"] = "Observed"
    return out


def thin(df: pd.DataFrame, max_points: int = MAX_HISTORY_POINTS) -> tuple[pd.DataFrame, int]:
    if len(df) <= max_points:
        return df, 1
    k = math.ceil(len(df) / max_points)
    return df.iloc[::k], k


def level_key(lv) -> str:
    """'0.8' -> '80' (used in column names; the level itself is never changed)."""
    return f"{float(lv) * 100:g}".replace(".", "p")


def forecast_frame(series: dict) -> pd.DataFrame:
    rows = []
    for pt in series.get("forecast") or []:
        r = {"timestamp": pd.to_datetime(pt["timestamp"]), "value": pt["point"], "kind": "Forecast", "step": pt["step"]}
        for lv, (lo, hi) in (pt.get("intervals") or {}).items():
            r[f"lo_{level_key(lv)}"], r[f"hi_{level_key(lv)}"] = lo, hi        # column names without dots (chart field names)
        rows.append(r)
    return pd.DataFrame(rows)


def interval_levels(series: dict) -> list[str]:
    if (series.get("uncertainty") or {}).get("status") != "available":
        return []
    lv = set()
    for pt in series.get("forecast") or []:
        lv |= set((pt.get("intervals") or {}).keys())
    return sorted(lv, key=float)


def candidates_table(series: dict) -> pd.DataFrame:
    rows = []
    for c in series.get("candidates") or []:
        rows.append({"Model": c.get("model"), "Status": c.get("status"), **{lab: c.get(k) for k, lab in VALIDATION_LABELS.items()},
                     "Evidence level": c.get("evidence_level"), "Note": c.get("reason") or "", "Selected": "yes" if c.get("model") == series.get("selected_model") else ""})
    return pd.DataFrame(rows)


def all_issues(resp: dict, series: dict | None = None) -> list[dict]:
    out = list(resp.get("issues") or [])
    if series:
        out += list(series.get("issues") or [])
    return out
