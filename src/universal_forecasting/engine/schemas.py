"""Public request / response schemas of the forecasting engine (typed, explicitly validated, JSON-serialisable).

Numbers in a ForecastResponse are produced by the numerical backend only. A future language layer may *explain*
them (it can read `ForecastResponse.to_dict()`), but must never compute, overwrite or invent numeric fields.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

ENGINE_VERSION = "phase4a-v1.0"
SCHEMA_VERSION = "forecast-response-v1"


class Severity(str, Enum):
    ERROR = "error"          # blocking: no forecast is produced for the affected scope
    WARNING = "warning"      # forecast produced; the user should know
    INFO = "info"


class Status(str, Enum):
    OK = "ok"                # every requested series forecast
    PARTIAL = "partial"      # some series failed (each failure carries its issues)
    ERROR = "error"          # nothing forecast


@dataclass
class Issue:
    code: str                # stable machine-readable code, e.g. "DUPLICATE_TIMESTAMPS"
    severity: str
    message: str             # human-readable, actionable
    hint: str = ""           # what the user can do about it
    scope: str = "request"   # "request" | "series:<group>|<target>" | "column:<name>"
    details: dict = field(default_factory=dict)


class RequestError(ValueError):
    def __init__(self, issues: list[Issue]):
        super().__init__("; ".join(i.message for i in issues))
        self.issues = issues


HORIZON_UNITS = ("steps",)   # 4A: the horizon counts steps of the (declared or inferred) regular frequency


@dataclass
class ForecastRequest:
    data: pd.DataFrame
    timestamp_column: str
    target_columns: list[str]
    horizon: int
    horizon_unit: str = "steps"
    group_column: str | None = None
    frequency: str | None = None                       # pandas offset alias, e.g. "h", "D"; inferred if None (and reported)
    known_future_covariates: list[str] = field(default_factory=list)
    allowed_models: list[str] | None = None            # None = every eligible model in the registry
    seasonal_period: int | None = None                 # override; otherwise derived from the frequency (reported)
    interval_levels: tuple[float, ...] = (0.8, 0.95)
    max_missing_fraction: float = 0.05                 # above this, missing grid points are a blocking error
    validation_fraction: float = 0.2

    def validate(self) -> None:
        bad: list[Issue] = []
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            bad.append(Issue("EMPTY_DATA", Severity.ERROR, "data must be a non-empty pandas DataFrame", "Pass the table to forecast."))
        else:
            cols = set(self.data.columns)
            if self.timestamp_column not in cols:
                bad.append(Issue("MISSING_TIMESTAMP_COLUMN", Severity.ERROR, f"timestamp column '{self.timestamp_column}' not found",
                                 f"Choose one of: {sorted(map(str, cols))[:20]}"))
            if not self.target_columns:
                bad.append(Issue("NO_TARGET", Severity.ERROR, "at least one target column must be named explicitly",
                                 "The engine never guesses targets from numeric columns."))
            for c in self.target_columns:
                if c not in cols:
                    bad.append(Issue("MISSING_TARGET_COLUMN", Severity.ERROR, f"target column '{c}' not found", "Check the column name.", f"column:{c}"))
                elif c == self.timestamp_column or c == self.group_column:
                    bad.append(Issue("INVALID_TARGET", Severity.ERROR, f"'{c}' cannot be both a target and the timestamp/group column", "", f"column:{c}"))
            if self.group_column is not None and self.group_column not in cols:
                bad.append(Issue("MISSING_GROUP_COLUMN", Severity.ERROR, f"group column '{self.group_column}' not found", ""))
            for c in self.known_future_covariates:
                if c not in cols:
                    bad.append(Issue("MISSING_COVARIATE_COLUMN", Severity.ERROR, f"covariate '{c}' not found", "", f"column:{c}"))
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 1:
            bad.append(Issue("INVALID_HORIZON", Severity.ERROR, "horizon must be a positive integer number of steps", ""))
        if self.horizon_unit not in HORIZON_UNITS:
            bad.append(Issue("UNSUPPORTED_HORIZON_UNIT", Severity.ERROR, f"horizon_unit '{self.horizon_unit}' is not supported",
                             "Phase 4A supports horizon_unit='steps' of a regular frequency; calendar/business-day semantics are not inferred."))
        if self.frequency is not None:
            try:
                pd.tseries.frequencies.to_offset(self.frequency)
            except ValueError:
                bad.append(Issue("INVALID_FREQUENCY", Severity.ERROR, f"frequency '{self.frequency}' is not a pandas offset alias", "Use e.g. 'h', 'D', '15min'."))
        if any(not (0 < lv < 1) for lv in self.interval_levels):
            bad.append(Issue("INVALID_INTERVAL_LEVEL", Severity.ERROR, "interval levels must be in (0, 1)", ""))
        if not (0.05 <= self.validation_fraction <= 0.5):
            bad.append(Issue("INVALID_VALIDATION_FRACTION", Severity.ERROR, "validation_fraction must be in [0.05, 0.5]", ""))
        if self.seasonal_period is not None and (not isinstance(self.seasonal_period, int) or self.seasonal_period < 2):
            bad.append(Issue("INVALID_SEASONAL_PERIOD", Severity.ERROR, "seasonal_period must be an integer >= 2", ""))
        if bad:
            raise RequestError(bad)

    def echo(self) -> dict:
        d = {
            "timestamp_column": self.timestamp_column, "target_columns": list(self.target_columns), "horizon": self.horizon, "horizon_unit": self.horizon_unit,
            "group_column": self.group_column, "frequency": self.frequency, "known_future_covariates": list(self.known_future_covariates),
            "allowed_models": self.allowed_models, "seasonal_period": self.seasonal_period, "interval_levels": list(self.interval_levels),
            "max_missing_fraction": self.max_missing_fraction, "validation_fraction": self.validation_fraction}
        d["n_rows"] = int(len(self.data)) if isinstance(self.data, pd.DataFrame) else 0
        return d


@dataclass
class ForecastPoint:
    step: int
    timestamp: str
    point: float
    intervals: dict = field(default_factory=dict)       # {"0.8": [lo, hi], ...} only when uncertainty.status == "available"


@dataclass
class CandidateResult:
    model: str
    status: str                       # evaluated | failed | ineligible | excluded_by_user
    reason: str = ""
    val_mae: float | None = None
    val_rmse: float | None = None
    rel_mae_vs_naive: float | None = None
    mase: float | None = None
    per_step_mae: list | None = None
    n_origins: int = 0
    evidence_level: str = ""


@dataclass
class SeriesResult:
    group: str | None
    target: str
    status: str                       # ok | error
    frequency: str | None = None
    frequency_source: str = ""        # declared | inferred
    seasonal_period: int | None = None
    history: dict = field(default_factory=dict)
    forecast_origin: str | None = None
    training_boundary: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    selected_model: str | None = None
    selection_reason: str = ""
    baseline_model: str = "naive"
    candidates: list = field(default_factory=list)
    forecast: list = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    limitations: list = field(default_factory=list)


@dataclass
class ForecastResponse:
    status: str
    engine_version: str = ENGINE_VERSION
    schema_version: str = SCHEMA_VERSION
    request: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    series: list = field(default_factory=list)
    registry: list = field(default_factory=list)
    numeric_provenance: str = "all numbers computed by the numerical backend (no language model involved)"

    def to_dict(self) -> dict:
        return _clean(asdict(self))


def _clean(o: Any):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _clean(o.item())
        except Exception:                                     # noqa: BLE001
            return str(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o
