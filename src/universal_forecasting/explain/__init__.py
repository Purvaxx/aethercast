"""Grounded explanation layer (Phase 4B) on top of the Phase 4A forecasting engine.

    from src.universal_forecasting.explain import ExplanationRequest, explain
    out = explain(ExplanationRequest(response=engine_response.to_dict(), question="Why was this model selected?"))

Works with no API key (deterministic template). With a provider, language-model wording is used only after it passes
the grounding validator; otherwise the deterministic explanation is returned.
"""
from .api import explain
from .schemas import EXPLAIN_VERSION, ExplanationRequest, ExplanationResponse

__all__ = ["explain", "ExplanationRequest", "ExplanationResponse", "EXPLAIN_VERSION"]
