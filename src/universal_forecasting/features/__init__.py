"""Phase 3A universal feature engine: past-only, per-entity features with declared availability.

Modules: base (spec/roles/availability), lags, rolling, returns, calendar, policy (dataset configs + safe set),
engine (computation, chunked writing, causality check).
"""
FEATURE_VERSION = "phase3a-v1.0"
