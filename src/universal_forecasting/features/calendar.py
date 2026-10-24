"""Calendar features (KNOWN-FUTURE covariates): deterministic functions of a timestamp in the dataset's LOCAL clock.

Phase 2 stores naive timestamps in each dataset's declared clock (PJM: US/Eastern wall clock, NSE: Asia/Kolkata,
crypto: UTC, ...). `to_local` converts tz-aware input to that clock and leaves naive input as is (already local).
Cyclical encodings use sin/cos so that 23h and 0h are close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Feature, FeatureSpec, Leakage, MissingPolicy, Role

FIELDS = ("hour", "minute_of_day", "dayofweek", "dayofyear", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
          "month_sin", "month_cos", "minutes_since_open")


def to_local(ts: pd.Series, tz: str | None) -> pd.Series:
    ts = pd.to_datetime(ts)
    if getattr(ts.dt, "tz", None) is not None:
        if not tz:
            raise ValueError("tz-aware timestamps need the dataset timezone")
        return ts.dt.tz_convert(tz).dt.tz_localize(None)
    return ts


def calendar_frame(ts: pd.Series, fields, tz: str | None = None, session_open: str | None = None) -> pd.DataFrame:
    t = to_local(ts, tz)
    out = {}
    for f in fields:
        if f == "hour": out[f] = t.dt.hour
        elif f == "minute_of_day": out[f] = t.dt.hour * 60 + t.dt.minute
        elif f == "dayofweek": out[f] = t.dt.dayofweek
        elif f == "dayofyear": out[f] = t.dt.dayofyear
        elif f == "month": out[f] = t.dt.month
        elif f == "is_weekend": out[f] = (t.dt.dayofweek >= 5).astype("int8")
        elif f == "hour_sin": out[f] = np.sin(2 * np.pi * t.dt.hour / 24)
        elif f == "hour_cos": out[f] = np.cos(2 * np.pi * t.dt.hour / 24)
        elif f == "dow_sin": out[f] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
        elif f == "dow_cos": out[f] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
        elif f == "doy_sin": out[f] = np.sin(2 * np.pi * (t.dt.dayofyear - 1) / 365.25)
        elif f == "doy_cos": out[f] = np.cos(2 * np.pi * (t.dt.dayofyear - 1) / 365.25)
        elif f == "month_sin": out[f] = np.sin(2 * np.pi * (t.dt.month - 1) / 12)
        elif f == "month_cos": out[f] = np.cos(2 * np.pi * (t.dt.month - 1) / 12)
        elif f == "minutes_since_open":
            h, m = map(int, (session_open or "00:00").split(":"))
            out[f] = t.dt.hour * 60 + t.dt.minute - (h * 60 + m)
        else:
            raise ValueError(f"unknown calendar field {f}")
    return pd.DataFrame({f"cal_{k}": v.astype("float64") for k, v in out.items()}, index=ts.index)


@dataclass
class CalendarFeature(Feature):
    fields: tuple[str, ...]
    timezone: str = ""
    session_open: str | None = None
    role: Role = Role.KNOWN_FUTURE_COVARIATE

    def compute(self, df, g):
        cf = calendar_frame(df["timestamp"], self.fields, None, self.session_open)   # Phase 2 timestamps are already local
        return {c: cf[c] for c in cf.columns}

    def specs(self):
        return [FeatureSpec(f"cal_{f}", ["timestamp"], f"calendar {f} of the row timestamp in the dataset's local clock ({self.timezone})", 0,
                            "known in advance for any timestamp (deterministic)", self.role.value, Leakage.SAFE.value, MissingPolicy.PROPAGATE.value,
                            {"timezone": self.timezone, "session_open": self.session_open}) for f in self.fields]
