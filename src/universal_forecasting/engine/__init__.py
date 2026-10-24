"""Universal forecasting engine (Phase 4A): typed request/response, dataset profiling, model registry with evidence
levels, leakage-safe chronological selection with a naive fallback, and validation-residual intervals.

    from src.universal_forecasting.engine import ForecastRequest, forecast
    response = forecast(ForecastRequest(data=df, timestamp_column="ts", target_columns=["y"], horizon=24))
    response.to_dict()      # JSON-serialisable; every number comes from the numerical backend
"""
from .api import forecast
from .schemas import ENGINE_VERSION, ForecastRequest, ForecastResponse, Issue, RequestError, Severity, Status

__all__ = ["forecast", "ForecastRequest", "ForecastResponse", "Issue", "RequestError", "Severity", "Status", "ENGINE_VERSION"]
