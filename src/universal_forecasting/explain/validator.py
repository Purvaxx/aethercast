"""Grounding validator for language-model output. A model answer is used only if EVERY check passes; otherwise the
deterministic explanation is returned and the rejection reasons are recorded.

Checks
  schema        JSON object with the required string fields and nothing structural missing.
  numbers       every number in the text must equal an evidence value (engine field or documented derivation) when that
                value is rounded to the number of decimals written (a trailing % may also match value*100);
  timestamps    every date / date-time written must be one of the evidence timestamps;
  claims        no significance, guarantee, proof, causation, correctness-probability, confidence-score or "test"
                performance wording unless negated ("not guaranteed" is fine);
  models        the model named as selected must be the engine's selected model; the PJM MLP may not be presented as
                validated/recommended; no unknown model names;
  references    cited evidence paths must exist in the evidence index.
"""
from __future__ import annotations

import json
import re

from .evidence import Evidence

TEXT_FIELDS = ("summary", "trend_interpretation", "model_selection_explanation", "uncertainty_explanation")
NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?\s?%?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
NEG = re.compile(r"\b(not|no|never|cannot|can't|doesn't|don't|isn't|aren't|without|nor|neither|unavailable|n't)\b[^.;:]{0,60}$", re.I)
FORBIDDEN = [("statistical_significance", r"statistically significant|statistical significance|p-?value|significantly (better|worse|outperform)"),
             ("guarantee", r"guarantee[sd]?|certain(ly)? will|will definitely|is sure to|bound to"),
             ("proof", r"\bprove[sdn]?\b|proven|proof"),
             ("causation", r"\bcaus(e|es|ed|ing)\b|because of the (weather|market|economy)|driven by|due to (the )?(economy|market|weather|news)"),
             ("correctness_probability", r"probability (that|of) (the forecast|it|this) (is|being) (correct|right|accurate)|chance (that )?(the forecast|it) is (correct|right)|\d+\s?% (chance|probability|likely|confident|confidence)|confidence score|reliability score"),
             ("test_performance", r"\btest (set|data|performance|accuracy|period|results?|error|mae|metric)|out-of-sample test|held-out test|on the test\b"),
             ("universal_superiority", r"best model overall|universally best|always (the )?best|best in general|superior model|outperforms all"),
             ("extra_analysis", r"i (re-?ran|recomputed|recalculated|calculated|computed|re-?trained|trained|simulated|tested)|additional analysis (shows|showed)|i ran")]
PROMOTE = re.compile(r"(validated|confirmed|proven|recommended|production-ready|best)", re.I)
KNOWN_OTHER_MODELS = re.compile(r"\b(arima|sarima|prophet|lstm|xgboost|lightgbm|transformer|tft|n-?beats|chronos|timesfm)\b", re.I)


def parse(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("model output is not text")
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def _negated(text: str, start: int) -> bool:
    return bool(NEG.search(text[max(0, start - 80): start]))


def check_numbers(text: str, ev: Evidence) -> list[str]:
    bad = []
    stamps = set()
    for t in ev.timestamps():
        stamps |= {t, t.replace("T", " "), t[:10], t[:16], t[:16].replace("T", " ")}
    work = text
    for m in DATE.finditer(text):
        if m.group(0) not in stamps:
            bad.append(f"timestamp not in evidence: {m.group(0)}")
        work = work.replace(m.group(0), " ")
    nums = list(ev.numbers().values())
    for m in NUM.finditer(work):
        tok = m.group(0).strip()
        pct = tok.endswith("%")
        core = tok.rstrip("%").strip().replace(",", "").replace("−", "-")
        try:
            x = float(core)
        except ValueError:
            continue
        dec = len(core.split(".")[1].split("e")[0].split("E")[0]) if "." in core else 0
        tol = 0.5 * 10 ** (-dec) + 1e-12
        ok = any(abs(round(v, dec) - x) <= 1e-9 or abs(v - x) <= tol * 1e-6 for v in nums) or any(abs(round(abs(v), dec) - x) <= 1e-9 for v in nums)
        if not ok and pct:
            ok = any(abs(round(v * 100, dec) - x) <= 1e-9 for v in nums)
        if not ok:
            bad.append(f"number not in evidence: {tok}")
    return bad


def check_claims(text: str) -> list[str]:
    bad = []
    for name, pat in FORBIDDEN:
        for m in re.finditer(pat, text, re.I):
            if not _negated(text, m.start()):
                bad.append(f"unsupported claim ({name}): '{m.group(0)}'")
                break
    return bad


def check_models(text: str, ev: Evidence) -> list[str]:
    bad = []
    sel = ev.facts["selected_model"]
    names = [c["model"] for c in ev.facts["candidates"]]
    for m in re.finditer(r"\b([a-z_0-9]+)\b[^.]{0,30}\b(was|is|has been)\s+(selected|chosen)\b", text, re.I):
        n = m.group(1).lower()
        if n in names and n != sel and not _negated(text, m.end() - len(m.group(3))):
            bad.append(f"wrong selected model: {n} (engine selected {sel})")
    for m in re.finditer(r"mlp[_a-z0-9]*", text, re.I):
        window = text[max(0, m.start() - 80): m.end() + 80]
        p = PROMOTE.search(window)
        if p and not re.search(r"\b(not|never|un|no)\w*\b[^.]{0,40}" + re.escape(p.group(0)), window, re.I):
            bad.append("PJM MLP presented as validated/recommended (it is exploratory; primary endpoint not met)")
    for m in KNOWN_OTHER_MODELS.finditer(text):
        bad.append(f"model not in the engine response: {m.group(0)}")
    return bad


def validate(raw: str, ev: Evidence, want_answer: bool) -> tuple[dict | None, list[str]]:
    try:
        obj = parse(raw)
    except Exception as ex:                                   # noqa: BLE001
        return None, [f"malformed output: {type(ex).__name__}"]
    reasons = []
    need = TEXT_FIELDS + (("answer",) if want_answer else ())
    for k in need:
        if not isinstance(obj.get(k), str) or not obj[k].strip():
            reasons.append(f"missing or non-text field: {k}")
    extra = set(obj) - set(TEXT_FIELDS) - {"answer", "evidence_used"}
    if extra:
        reasons.append(f"unexpected fields: {sorted(extra)}")
    if reasons:
        return None, reasons
    used = obj.get("evidence_used") or {}
    if not isinstance(used, dict) or any(not isinstance(v, list) for v in used.values()):
        reasons.append("evidence_used is not a {section: [paths]} object")
    else:
        for sec, paths in used.items():
            for p in paths:
                if p not in ev.index:
                    reasons.append(f"evidence path does not exist: {str(p)[:80]}")
    for k in need:
        t = obj[k]
        if len(t) > 6000:
            reasons.append(f"{k} too long")
        reasons += [f"{k}: {r}" for r in check_numbers(t, ev) + check_claims(t) + check_models(t, ev)]
    return (obj if not reasons else None), reasons
