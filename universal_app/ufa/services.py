"""Service boundary between the interface and the forecasting backend.

The interface talks ONLY to a ForecastService:
    validate(dataset_frame, config)            -> list of issues ({code, severity, message, hint})
    run_forecast(dataset_frame, config)        -> Phase 4A response dict  (or raises ServiceUnavailable)
    explain(response, question, style)         -> Phase 4B response dict  (or raises ServiceUnavailable)

Implementations
    EngineService         NORMAL MODE (default since Phase 6). run_forecast() hands the uploaded frame and the user's
                          choices to the real Phase 4A engine and returns its response unchanged; explain() hands that
                          response to the Phase 4B layer on its deterministic path (use_llm=False: no language model,
                          no API key). The adapter computes nothing: it only checks that the response is the
                          answer to the request that was sent (engine_contract_problems).
    NotConnectedService   UFA_SERVICE=not_connected. Request validation is real; forecast execution and explanations
                          are refused and say so; nothing is fabricated.
    DemoService           replays ONE stored, genuine Phase 4A response that was computed by the real engine on the
                          bundled synthetic sample file, and only for that exact file and configuration. Explanations
                          come from the Phase 4B deterministic path (use_llm=False: no language model, no API key).
                          Every screen shows a DEMO banner.
    MockService           injectable canned responses for UI tests.

No service ever falls back to another one: an engine failure is reported as a failure, never replaced by the demo
response. The services log codes, shapes and timings only, never dataset contents, questions or credentials.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import datasets as DS  # noqa: E402
from src.universal_forecasting.engine.registry import REGISTRY  # noqa: E402  (schemas/registry by reference; no numerical logic copied)
from src.universal_forecasting.engine.schemas import ForecastRequest, RequestError  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
SAMPLE_CSV = SAMPLE_DIR / "synthetic_hourly_demo.csv"
SAMPLE_RESPONSE = SAMPLE_DIR / "synthetic_hourly_demo_response.json"
SAMPLE_META = SAMPLE_DIR / "synthetic_hourly_demo_meta.json"

FREQUENCY_OPTIONS = {"Infer from the timestamps": None, "Every minute (min)": "min", "Every 5 minutes (5min)": "5min", "Every 15 minutes (15min)": "15min",
                     "Every 30 minutes (30min)": "30min", "Hourly (h)": "h", "Daily (D)": "D", "Business days (B)": "B", "Weekly (7D)": "7D",
                     "Monthly, month start (MS)": "MS", "Quarterly, quarter start (QS)": "QS"}
ROUTED_MODELS = [m.name for m in sorted(REGISTRY.values(), key=lambda s: s.cost_rank) if m.routed]
# ---- form and service settings: THE single source for the interface (none of these changes a forecast number)
MAX_HORIZON_UI = 1000                               # largest horizon, in steps, the form accepts
LARGE_ROWS = 250_000                                # from this many rows the "long run, cannot be cancelled" notice is shown
SERVICE_ENV = "UFA_SERVICE"                         # environment variable that selects the mode
SERVICE_MODES = ("engine", "demo", "not_connected")
DEFAULT_SERVICE_MODE = "engine"                     # normal mode: the real Phase 4A engine
HOST, PORT = "localhost", 8503                      # where launch.py serves the app (literals there; a test keeps them equal)
LARGE_ROWS_NOTE = ("A forecast on a file this size takes a while and cannot be cancelled or timed out once started. On the development computer (Phase 7 "
                   "measurements, minute-level data) 500,000 rows took about 45 seconds and 2,000,000 rows about 3 minutes; your computer may differ. "
                   "Keep this tab open until it finishes, or upload a shorter extract.")
log = logging.getLogger("ufa.service")             # local development log (stderr of the app process)
if not log.handlers:                               # codes, shapes and timings only; see the module docstring
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s ufa.service %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------- progress events (real pipeline milestones, no timers)
# The interface shows progress only from events emitted here. Each event fires at the moment the corresponding real
# step starts or finishes; nothing is estimated. Engine events come from pass-through probes placed around four engine
# functions at run time (the engine's source files are not modified and its results are byte-identical): the probes
# call the original function with the same arguments and return its result unchanged.
import functools  # noqa: E402
import threading  # noqa: E402

def _sink():
    """The per-thread slot that holds the current listener. It lives on the engine's API module, not on this module, so
    that the engine probes and this code always share one slot even if Streamlit reloads this file while the app runs."""
    import src.universal_forecasting.engine.api as api
    slot = getattr(api, "_ufa_progress_slot", None)
    if slot is None:
        slot = api._ufa_progress_slot = threading.local()
    return slot


class progress_events:
    """with progress_events(callback): ... -> callback(event: str, info: dict) for every milestone on this thread."""
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        self.previous = getattr(_sink(), "callback", None)
        _sink().callback = self.callback
        return self

    def __exit__(self, *exc):
        _sink().callback = self.previous
        return False


def emit(event: str, info: dict | None = None, **more) -> None:
    cb = getattr(_sink(), "callback", None)
    if cb is None:
        return
    try:
        cb(event, {**(info or {}), **more})
    except Exception as ex:                                       # noqa: BLE001  (a display problem must never disturb a forecast)
        log.warning("progress display failed on %s (%s)", event, type(ex).__name__)


def _install_engine_probes() -> None:
    """Idempotent. Wraps profile / evaluate / intervals / models.build so that they report when they start and finish."""
    import src.universal_forecasting.engine.api as api
    if getattr(api, "_ufa_probes_installed", False):
        return
    models = api.M        # the engine's own model module, reached through its API module; only its build() is wrapped, nothing is computed here

    def probe(module, name, before=None, after=None):
        original = getattr(module, name)

        def tell(event, info):                                    # self-contained: depends only on the shared slot, not on this module's globals
            cb = getattr(getattr(api, "_ufa_progress_slot", None), "callback", None)
            if cb is not None:
                try:
                    cb(event, info)
                except Exception:                                 # noqa: BLE001  (a display problem must never disturb a forecast)
                    pass

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if before:
                tell(*before(args, kwargs))
            result = original(*args, **kwargs)
            if after:
                tell(*after(result))
            return result
        setattr(module, name, wrapper)

    probe(api, "profile", before=lambda a, k: ("profile_start", {}), after=lambda r: ("profile_done", {"n_series": len(r[2]), "n_usable": sum(bool(p.ok) for p in r[2])}))
    probe(api, "evaluate", before=lambda a, k: ("evaluate_start", {}), after=lambda r: ("evaluate_done", {"selected": getattr(r, "selected", None)}))
    probe(api, "intervals", before=lambda a, k: ("intervals_start", {}))
    probe(models, "build", before=lambda a, k: ("model", {"name": a[0] if a else k.get("name")}))
    api._ufa_probes_installed = True


ENGINE_STAGES = [("data", "Dataset in memory"), ("validate", "Validating the request: timestamps, target and settings"),
                 ("prepare", "Engine: checking values and gaps, preparing the time series"), ("models", "Engine: scoring candidate models on the validation period"),
                 ("final", "Engine: final fit, forecast and prediction intervals"), ("finish", "Checking the response and writing the explanation")]
SERVICE_STAGES = [("data", "Dataset in memory"), ("validate", "Validating the request: timestamps, target and settings"),
                  ("service", "Asking the forecast service"), ("finish", "Checking the response and writing the explanation")]
PROGRESS_BASIS = "Progress counts completed pipeline stages. It is not a time estimate."


class RunProgress:
    """State of one forecast run, driven ONLY by real events (see emit). No clock moves it forward: the percentage is a
    fixed milestone value for each stage that has actually been reached, so it can stand still for as long as a stage
    takes, never runs ahead of the work, and reaches 100 only on the 'done' event."""

    def __init__(self, mode: str = "engine", n_models: int = 1):
        self.stages = list(ENGINE_STAGES if mode == "engine" else SERVICE_STAGES)
        self.keys = [k for k, _ in self.stages]
        self.n_models = max(int(n_models), 1)
        self.state, self.stage, self.percent, self.detail = "running", "data", 0, ""
        self.n_series, self.series_i, self.model_i, self.in_evaluate = 1, 0, 0, False
        self.started, self.seconds, self.events = {}, {}, []
        self._enter("data")

    # -- helpers
    def _enter(self, key: str):
        if key not in self.keys:
            return
        for k in self.keys[: self.keys.index(key)]:               # every earlier stage is over
            self._leave(k)
        self.stage = key
        self.started.setdefault(key, time.perf_counter())

    def _leave(self, key: str):
        if key in self.started and key not in self.seconds:
            self.seconds[key] = time.perf_counter() - self.started[key]

    def _to(self, pct: float):
        self.percent = max(self.percent, min(int(round(pct)), 99))    # 100 is reserved for 'done'

    def _span(self):                                              # engine block 30..86, shared equally between series
        width = 56.0 / max(self.n_series, 1)
        return 30.0 + width * max(self.series_i - 1, 0), width

    def exclude(self, seconds: float) -> None:
        """A display pause (not work) happened during the current stage: keep it out of that stage's measured time."""
        if self.stage in self.started and self.stage not in self.seconds:
            self.started[self.stage] += max(float(seconds), 0.0)

    @property
    def index(self) -> int:
        return self.keys.index(self.stage)

    def status(self, key: str) -> str:
        if self.state == "done" or self.keys.index(key) < self.index:
            return "done"
        if key == self.stage:
            return "failed" if self.state == "failed" else "running"
        return "waiting"

    # -- the only way the state changes
    def on(self, event: str, info: dict | None = None) -> None:
        info = info or {}
        if self.state != "running":
            return
        self.events.append(event)
        if event == "data_ready":
            self.detail = f"{info.get('rows', 0):,} rows, {info.get('columns', 0)} columns"
            self._to(8)
        elif event == "validate_start":
            self._enter("validate"); self.detail = ""
        elif event == "validate_done":
            self._to(18)
        elif event == "engine_start":
            self._enter("prepare" if "prepare" in self.keys else "service"); self.detail = ""
            self._to(20)
        elif event == "profile_done":
            self.n_series = max(int(info.get("n_series") or 1), 1)
            self.detail = f"{self.n_series} series prepared" if self.n_series > 1 else "series prepared"
            self._to(30)
        elif event == "evaluate_start":
            self.series_i += 1; self.model_i = 0; self.in_evaluate = True
            self._enter("models")
            self.detail = f"series {self.series_i} of {self.n_series}" if self.n_series > 1 else ""
            self._to(self._span()[0])
        elif event == "model":
            base, width = self._span()
            if self.in_evaluate:
                self.model_i += 1
                self.detail = (f"series {self.series_i} of {self.n_series}, " if self.n_series > 1 else "") + f"model {min(self.model_i, self.n_models)} of {self.n_models}: {info.get('name')}"
                self._to(base + 0.75 * width * (min(self.model_i, self.n_models) - 1) / self.n_models)
            else:
                self.detail = f"final fit: {info.get('name')}"
        elif event == "evaluate_done":
            self.in_evaluate = False
            if self.n_series == 1:
                self._enter("final")
            self.detail = f"selected {info.get('selected')}" if info.get("selected") else ""
            base, width = self._span()
            self._to(base + 0.75 * width)
        elif event == "intervals_start":
            base, width = self._span()
            self.detail = "prediction intervals from validation residuals"
            self._to(base + 0.9 * width)
        elif event == "engine_done":
            self._enter("finish"); self.detail = "response received"
            self._to(86)
        elif event == "response_checked":
            self._to(90)
        elif event == "explain_start":
            self.detail = "explanation"
        elif event == "explain_done":
            self._to(96)
        elif event == "done":
            for k in self.keys:
                self._leave(k)
            self.state, self.percent, self.detail = "done", 100, "forecast ready"
        elif event == "failed":
            self._leave(self.stage)
            self.state, self.detail = "failed", str(info.get("code") or "stopped")

    def snapshot(self) -> dict:
        return {"state": self.state, "stage": self.stage, "percent": self.percent, "detail": self.detail,
                "stages": [{"key": k, "label": lab, "status": self.status(k), "seconds": self.seconds.get(k)} for k, lab in self.stages]}


def where(ex: BaseException, limit: int = 8) -> str:
    """Where an exception was raised, as 'file:line function' steps. Deliberately WITHOUT the exception's message: a
    message can quote a value from the uploaded data or a credential, and the log must hold neither."""
    import traceback
    frames = traceback.extract_tb(ex.__traceback__)[-limit:]
    return " > ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames) or "unknown"


class ServiceUnavailable(RuntimeError):
    """kind: 'unavailable' = the service declined (nothing went wrong); 'error' = the backend failed."""
    def __init__(self, code: str, message: str, kind: str = "unavailable"):
        super().__init__(message)
        self.code, self.message, self.kind = code, message, kind


@dataclass
class ForecastConfig:
    timestamp_column: str | None = None
    target_column: str | None = None
    horizon: int | None = None
    group_column: str | None = None
    frequency: str | None = None                 # None = infer
    allowed_models: list | None = None           # None = all routed models
    confirmed: bool = False                      # the user ticked "I confirm these choices"

    def to_request(self, frame: pd.DataFrame) -> ForecastRequest:
        return ForecastRequest(data=frame, timestamp_column=self.timestamp_column or "", target_columns=[self.target_column] if self.target_column else [],
                               horizon=self.horizon if self.horizon is not None else 0, group_column=self.group_column, frequency=self.frequency,
                               allowed_models=self.allowed_models)


def ui_field_errors(frame: pd.DataFrame, cfg: ForecastConfig, tcheck=None) -> dict:
    """Errors keyed by control, shown next to it. Submission is disabled while this is non-empty."""
    e = {}
    if not cfg.timestamp_column:
        e["timestamp_column"] = "Choose the timestamp column."
    elif cfg.timestamp_column in frame.columns:
        tc = tcheck if tcheck is not None else DS.check_timestamps(frame, cfg.timestamp_column, cfg.group_column if cfg.group_column in frame.columns else None)
        if tc.blocking:
            e["timestamp_column"] = tc.blocking[0][1]
    if not cfg.target_column:
        e["target_column"] = "Choose the column to forecast."
    elif cfg.target_column == cfg.timestamp_column:
        e["target_column"] = "The target cannot be the timestamp column."
    elif cfg.target_column in frame.columns:
        tp = DS.target_problems(frame, cfg.target_column)
        if tp:
            e["target_column"] = tp[0][1]
    if cfg.group_column and cfg.group_column in (cfg.timestamp_column, cfg.target_column):
        e["group_column"] = "The series identifier must differ from the timestamp and target columns."
    elif cfg.group_column and cfg.group_column in frame.columns and DS.group_problems(frame, cfg.group_column):
        e["group_column"] = DS.group_problems(frame, cfg.group_column)[0][1]
    if cfg.horizon is None:
        e["horizon"] = "Enter the number of steps to forecast."
    elif not isinstance(cfg.horizon, int) or cfg.horizon < 1:
        e["horizon"] = "The horizon must be a whole number of at least 1 step."
    elif cfg.horizon > min(MAX_HORIZON_UI, max(len(frame) - 1, 1)):
        e["horizon"] = f"The horizon must be at most {min(MAX_HORIZON_UI, max(len(frame) - 1, 1))} steps for this dataset (fewer than its number of rows; interface cap {MAX_HORIZON_UI})."
    if cfg.allowed_models is not None and not cfg.allowed_models:
        e["allowed_models"] = "Select at least one model (the naive baseline is always included)."
    if not cfg.confirmed:
        e["confirmed"] = "Tick the confirmation above."
    return e


def request_key(dataset_sha256: str, cfg: ForecastConfig) -> str:
    """Identity of one forecast request (file contents + every setting). A stored result is shown only for the same key."""
    blob = json.dumps([dataset_sha256, cfg.timestamp_column, cfg.target_column, cfg.horizon, cfg.group_column, cfg.frequency, cfg.allowed_models], default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def engine_contract_problems(resp, cfg: ForecastConfig, n_rows: int) -> list[str]:
    """Is `resp` a Phase 4A response to exactly the request built from `cfg`? Compares named fields only; changes nothing."""
    if not isinstance(resp, dict):
        return ["the engine response is not an object"]
    p = []
    if resp.get("schema_version") != "forecast-response-v1":
        p.append(f"unexpected schema_version {resp.get('schema_version')!r}")
    if resp.get("status") not in ("ok", "partial", "error"):
        p.append("missing or unknown status")
    for k, t in (("series", list), ("issues", list), ("request", dict)):
        if not isinstance(resp.get(k), t):
            p.append(f"field '{k}' is missing or malformed")
    if p:
        return p
    echo = resp["request"]
    want = {"timestamp_column": cfg.timestamp_column, "target_columns": [cfg.target_column], "horizon": cfg.horizon, "horizon_unit": "steps",
            "group_column": cfg.group_column, "frequency": cfg.frequency, "allowed_models": cfg.allowed_models, "n_rows": n_rows}
    for k, v in want.items():
        if echo.get(k) != v:
            p.append(f"the response echoes a different request ({k})")
    for i, s in enumerate(resp["series"]):
        if not isinstance(s, dict):
            p.append(f"series {i} is malformed")
            continue
        if s.get("target") != cfg.target_column:
            p.append(f"series {i} is for a different target")
        if s.get("status") == "ok" and len(s.get("forecast") or []) != cfg.horizon:
            p.append(f"series {i} does not have exactly {cfg.horizon} forecast steps")
    if resp["status"] in ("ok", "partial") and not any(isinstance(s, dict) and s.get("status") == "ok" for s in resp["series"]):
        p.append("status says a forecast exists but no series has one")
    if resp["status"] == "error" and not resp["issues"] and not any(isinstance(s, dict) and s.get("issues") for s in resp["series"]):
        p.append("the engine reported an error without any reason")
    return p


def _explain_deterministic(response: dict, question, style: str) -> dict:
    """Phase 4B on its deterministic path. The response is passed as a deep copy, so nothing can alter the stored forecast."""
    from src.universal_forecasting.explain import ExplanationRequest, explain
    out = explain(ExplanationRequest(response=json.loads(json.dumps(response)), question=question or None, style=style, use_llm=False)).to_dict()
    log.info("explanation status=%s source=%s question=%s", out.get("status"), out.get("source"), bool(question))
    return out


def preflight_issues(frame, cfg) -> list[dict]:
    """Interface-side refusals, checked again here so that nothing reaches the engine by another route: inputs that the
    engine would accept but could read differently from what the file says (numeric or ambiguous timestamps, whole numbers
    too large for a double, rows without a series identifier). These are the interface's checks, not engine results."""
    out = []
    def add(pairs):
        out.extend({"code": c, "severity": "error", "message": m, "hint": "Fix the file and upload it again; the app does not rewrite your data."} for c, m in pairs)
    if cfg.timestamp_column in frame.columns:
        add(DS.check_timestamps(frame, cfg.timestamp_column, cfg.group_column if cfg.group_column in frame.columns else None).blocking)
    if cfg.target_column in frame.columns and cfg.target_column != cfg.timestamp_column:
        add([p for p in DS.target_problems(frame, cfg.target_column) if p[0] == "TARGET_NOT_EXACT"])     # the engine reports the other target problems itself
    if cfg.group_column and cfg.group_column in frame.columns:
        add(DS.group_problems(frame, cfg.group_column))
    return out


def _schema_issues(frame, cfg) -> list[dict]:
    try:
        cfg.to_request(frame).validate()            # REAL Phase 4A request-schema validation; runs no forecast
    except RequestError as ex:
        return [{"code": i.code, "severity": str(getattr(i.severity, "value", i.severity)), "message": i.message, "hint": i.hint} for i in ex.issues]
    return preflight_issues(frame, cfg)


class EngineService:
    """Normal mode: the real Phase 4A engine and the Phase 4B deterministic explanation. No fallback of any kind."""
    name, mode = "Forecasting engine (Phase 4A) and grounded explanation layer (Phase 4B, deterministic)", "engine"
    banner = None

    def validate(self, frame, cfg) -> list[dict]:
        return _schema_issues(frame, cfg)

    def run_forecast(self, frame, cfg) -> dict:
        from src.universal_forecasting.engine import forecast as engine_forecast     # the only place the interface calls the engine
        t0 = time.perf_counter()
        _install_engine_probes()                                  # progress reporting only; results are unchanged
        emit("engine_start")
        try:
            resp = engine_forecast(cfg.to_request(frame)).to_dict()
        except MemoryError:
            log.error("engine ran out of memory rows=%d cols=%d", len(frame), frame.shape[1])
            raise ServiceUnavailable("ENGINE_OUT_OF_MEMORY", "The forecasting engine ran out of memory on this dataset. No forecast was produced. "
                                     "Upload a smaller extract (fewer rows or columns).", "error") from None
        except Exception as ex:                                   # noqa: BLE001  (details go to the local log, not to the page)
            log.error("engine raised %s rows=%d cols=%d horizon=%s at %s", type(ex).__name__, len(frame), frame.shape[1], cfg.horizon, where(ex))
            raise ServiceUnavailable("ENGINE_EXCEPTION", f"The forecasting engine stopped with an internal error ({type(ex).__name__}). No forecast was produced. "
                                     "Check the dataset and settings and try again; the details are in the application log.", "error") from None
        emit("engine_done", status=resp.get("status") if isinstance(resp, dict) else None)
        problems = engine_contract_problems(resp, cfg, len(frame))
        if problems:
            log.error("engine response failed the adapter contract: %s", "; ".join(problems[:10]))
            raise ServiceUnavailable("MALFORMED_ENGINE_RESPONSE", "The forecasting engine returned a response that does not match this request, so it is not shown: "
                                     + "; ".join(problems[:5]) + ".", "error")
        log.info("forecast rows=%d cols=%d horizon=%s status=%s series=%d seconds=%.2f", len(frame), frame.shape[1], cfg.horizon, resp["status"], len(resp["series"]),
                 time.perf_counter() - t0)
        return resp

    def explain(self, response, question=None, style="concise") -> dict:
        return _explain_deterministic(response, question, style)


class NotConnectedService:
    name, mode = "Forecast engine not connected", "not_connected"
    banner = None

    def validate(self, frame, cfg) -> list[dict]:
        return _schema_issues(frame, cfg)

    def run_forecast(self, frame, cfg) -> dict:
        raise ServiceUnavailable("ENGINE_NOT_CONNECTED", "The forecasting engine is switched off in this mode (UFA_SERVICE=not_connected), so no forecast was computed. "
                                                         "Your dataset and settings passed the request checks. Nothing was estimated or invented.")

    def explain(self, response, question=None, style="concise") -> dict:
        raise ServiceUnavailable("EXPLANATION_NOT_CONNECTED", "Explanations need a forecast response, and the engine is switched off in this mode.")


class DemoService:
    """Replays the stored engine response for the bundled synthetic sample only."""
    name, mode = "Demo mode", "demo"
    banner = ("DEMO MODE: the forecast shown is a stored result that the real engine computed earlier on the bundled synthetic sample file "
              "(computer-generated data, not real measurements). Your own uploads are not forecast in this mode.")

    def __init__(self):
        self.meta = json.loads(SAMPLE_META.read_text(encoding="utf-8"))

    def validate(self, frame, cfg) -> list[dict]:
        return _schema_issues(frame, cfg)

    def run_forecast(self, frame, cfg) -> dict:
        key = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
        m = self.meta
        same_cfg = (cfg.timestamp_column == m["timestamp_column"] and cfg.target_column == m["target_column"] and cfg.horizon == m["horizon"]
                    and cfg.group_column is None and cfg.frequency in (None, m["frequency"]) and cfg.allowed_models in (None, ROUTED_MODELS))
        if key != m["frame_hash"] or not same_cfg:
            raise ServiceUnavailable("DEMO_ONLY_SAMPLE", "Demo mode can only replay the stored forecast for the bundled synthetic sample with its original settings "
                                     f"(timestamp '{m['timestamp_column']}', target '{m['target_column']}', horizon {m['horizon']}, frequency inferred, all models). "
                                     "No forecast was computed for this dataset or configuration.")
        return json.loads(SAMPLE_RESPONSE.read_text(encoding="utf-8"))

    def explain(self, response, question=None, style="concise") -> dict:
        return _explain_deterministic(response, question, style)


@dataclass
class MockService:
    """Injectable canned behaviour for UI tests (never used by the running app unless explicitly injected)."""
    forecast_response: dict | None = None
    explanation_response: dict | None = None
    validation_issues: list = field(default_factory=list)
    forecast_error: Exception | None = None
    explain_error: Exception | None = None
    name: str = "Mock service (tests)"
    mode: str = "mock"
    banner: str | None = "MOCK SERVICE: canned test responses, not a forecast."
    calls: list = field(default_factory=list)

    def validate(self, frame, cfg):
        self.calls.append("validate")
        return list(self.validation_issues)

    def run_forecast(self, frame, cfg):
        self.calls.append("run_forecast")
        if self.forecast_error:
            raise self.forecast_error
        if self.forecast_response is None:
            raise ServiceUnavailable("NO_RESPONSE", "The mock has no forecast response.")
        return json.loads(json.dumps(self.forecast_response))

    def explain(self, response, question=None, style="concise"):
        self.calls.append(("explain", question))
        if self.explain_error:
            raise self.explain_error
        if self.explanation_response is None:
            raise ServiceUnavailable("NO_EXPLANATION", "The mock has no explanation response.")
        return json.loads(json.dumps(self.explanation_response))


def make_service(mode: str | None):
    """None / '' / 'engine' -> normal mode (real engine). 'demo' and 'not_connected' must be asked for by name; anything else is an error, never a guess."""
    m = (mode or DEFAULT_SERVICE_MODE).strip().lower()
    if m not in SERVICE_MODES:
        raise ValueError(f"unknown UFA_SERVICE {mode!r}; use one of {SERVICE_MODES}")
    return {"engine": EngineService, "demo": DemoService, "not_connected": NotConnectedService}[m]()
