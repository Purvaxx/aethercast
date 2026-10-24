"""Feature specification, covariate roles, availability and leakage classes, and the per-entity helpers every
feature generator uses.

Row convention (all datasets): rows are sorted by (entity_id, timestamp) and a row's values are OBSERVED at its
timestamp (Phase 2 resampled rows are labelled at the END of their interval, so the label time is the time the value
becomes known). A feature value at row t may only depend on rows t, t-1, ... of the SAME entity. "Lag k" means k
observations back within the entity (equal to k steps on a regular grid; k sessions on a business-day calendar).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Role(str, Enum):
    TARGET_DERIVED = "target_derived"                    # computed from the target's own history
    OBSERVED_PAST_COVARIATE = "observed_past_covariate"  # other observed variables, known only up to t
    KNOWN_FUTURE_COVARIATE = "known_future_covariate"    # deterministic for any timestamp (calendar)
    UNKNOWN_FUTURE_COVARIATE = "unknown_future_covariate"  # raw covariate whose FUTURE values are unknown at t (never used beyond t)
    STATIC = "static"


class Leakage(str, Enum):
    SAFE = "safe"                        # point-in-time availability established
    LEAKAGE_PRONE = "leakage_prone"      # availability cannot be established -> excluded from the default safe set
    FORBIDDEN = "forbidden"              # uses future or hindsight information by construction


class MissingPolicy(str, Enum):
    PROPAGATE = "propagate"              # NaN in -> NaN out; rolling windows need min_periods valid values
    FFILL_PAST_LIMITED = "ffill_past_limited"  # carry the LAST PAST value forward at most `limit` rows (counted); never backward


@dataclass
class FeatureSpec:
    name: str
    source_columns: list[str]
    transformation: str
    lookback_rows: int                   # oldest row used, relative to t (0 = only row t)
    availability: str                    # rule stating when the value is known
    role: str
    leakage: str
    missing_policy: str
    params: dict = field(default_factory=dict)
    imputed_values: int = 0
    nan_values: int = 0

    def as_dict(self):
        return asdict(self)


class Feature:
    """Base class: `compute(df)` returns {name: Series aligned to df.index}; `specs()` describes them."""
    role = Role.TARGET_DERIVED

    def compute(self, df: pd.DataFrame, g) -> dict[str, pd.Series]:
        raise NotImplementedError

    def specs(self) -> list[FeatureSpec]:
        raise NotImplementedError


def entity_groups(df: pd.DataFrame):
    """groupby object on entity_id keeping row order (data must be sorted by entity, timestamp)."""
    return df.groupby("entity_id", sort=False, observed=True)


def check_sorted(df: pd.DataFrame) -> None:
    ent = df["entity_id"].to_numpy()
    ts = df["timestamp"].to_numpy()
    same = ent[1:] == ent[:-1]
    if (same & (ts[1:] <= ts[:-1])).any():
        raise ValueError("rows must be sorted by (entity_id, timestamp) with unique timestamps per entity")
    # an entity must occupy one contiguous block
    starts = pd.Series(ent).ne(pd.Series(ent).shift()).sum()
    if starts != pd.Series(ent).nunique():
        raise ValueError("entity rows are not contiguous")


def ffill_past(s: pd.Series, g, limit: int) -> tuple[pd.Series, int]:
    """Forward-fill within entity from PAST values only, at most `limit` consecutive rows. Returns (series, n_filled)."""
    filled = s.groupby(g.ngroup(), sort=False).ffill(limit=limit)
    return filled, int((s.isna() & filled.notna()).sum())


def stable_hash(obj) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def nan_count(s: pd.Series) -> int:
    return int(np.isnan(s.to_numpy(dtype="float64")).sum())
