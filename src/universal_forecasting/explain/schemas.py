"""Typed request / response of the explanation layer (Phase 4B).

The explanation layer never produces numbers. Every numeric value in an ExplanationResponse is copied from the
Phase 4A ForecastResponse (or derived from it by documented deterministic code in evidence.py), and the response says
which parts are engine facts and which are wording (template or language model).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

EXPLAIN_VERSION = "phase4b-v1.0"
EXPLAIN_SCHEMA = "explanation-response-v1"
STYLES = ("concise", "detailed")
MAX_QUESTION_CHARS = 1000


class ExplainInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ExplanationRequest:
    response: dict                       # ForecastResponse.to_dict() from the Phase 4A engine
    question: str | None = None
    style: str = "concise"
    group: str | None = None             # which series to explain when the response holds several
    target: str | None = None
    use_llm: bool | None = None          # None = use it only if a provider is configured; False = deterministic only

    def validate(self) -> None:
        r = self.response
        if hasattr(r, "to_dict"):
            self.response = r = r.to_dict()
        if not isinstance(r, dict):
            raise ExplainInputError("INVALID_ENGINE_RESPONSE", "response must be a Phase 4A ForecastResponse or its to_dict() form")
        if r.get("schema_version") != "forecast-response-v1":
            raise ExplainInputError("UNSUPPORTED_ENGINE_SCHEMA", f"unsupported engine schema_version {r.get('schema_version')!r}")
        for k, t in (("status", str), ("series", list), ("issues", list)):
            if not isinstance(r.get(k), t):
                raise ExplainInputError("MALFORMED_ENGINE_RESPONSE", f"engine response field '{k}' is missing or malformed")
        if self.style not in STYLES:
            raise ExplainInputError("INVALID_STYLE", f"style must be one of {STYLES}")
        if self.question is not None:
            if not isinstance(self.question, str):
                raise ExplainInputError("INVALID_QUESTION", "question must be text")
            if len(self.question) > MAX_QUESTION_CHARS:
                raise ExplainInputError("QUESTION_TOO_LONG", f"question longer than {MAX_QUESTION_CHARS} characters")


@dataclass
class QuestionResult:
    text: str | None = None
    intent: str | None = None            # meaning | model_selection | reliability | intervals | horizon_difficulty | limitations | baseline | trend | new_request | unsupported
    answerable: bool = False
    requires_new_forecast: bool = False
    new_request_hint: str = ""
    missing_evidence: list = field(default_factory=list)
    answer: str | None = None


@dataclass
class ExplanationResponse:
    status: str                          # ok | error
    error_code: str | None = None
    error_message: str | None = None
    explain_version: str = EXPLAIN_VERSION
    schema_version: str = EXPLAIN_SCHEMA
    source: str = "deterministic_template"      # deterministic_template | llm_validated | deterministic_fallback
    style: str = "concise"
    series: dict = field(default_factory=dict)
    summary: str = ""
    trend_interpretation: str = ""
    model_selection_explanation: str = ""
    uncertainty_explanation: str = ""
    question: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)      # copied verbatim from the engine
    warnings: list = field(default_factory=list)         # engine issues (code, severity, message), verbatim
    evidence_references: list = field(default_factory=list)   # [{section, paths:[...]}] -> paths into the engine response
    unavailable_evidence: list = field(default_factory=list)
    forecast: list = field(default_factory=list)         # engine forecast points, unchanged
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
