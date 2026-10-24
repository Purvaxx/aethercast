"""Run history for the Insights and Forecast Reports pages.

A run record holds what the engine returned for one request plus the small amount of context needed to show it again:
the file's name and fingerprint, the settings, the explanation text, and the LAST observed values of the target (at
most TAIL_POINTS per series) so a chart can be redrawn. The uploaded file itself is never stored.

Two stores with the same methods:
    SessionStore   records live in the browser session only (default when no folder is configured, e.g. in tests).
    FolderStore    one JSON file per run in a local folder on this computer (the launcher configures it), so reports
                   survive navigation, a refresh and a restart. Files are only ever created with new, unique names;
                   nothing is overwritten. Deleting is explicit, from the Reports page.
No secrets, environment variables or credentials are written.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from . import viewmodels as VM

RECORD_SCHEMA = "run-record-v1"
TAIL_POINTS = 400
STORE_ENV = "UFA_DATA_DIR"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(text, cap: int = 40) -> str:
    s = _SAFE.sub("_", str(text or "")).strip("._")
    return (s[:cap] or "x")


def make_record(ds, cfg, response: dict | None, explanation: dict | None, mode: str, seconds: float | None, error: dict | None = None, progress: dict | None = None) -> dict:
    """Build a record from a finished run (successful or failed). Values are copied from the engine response unchanged."""
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    rid = f"{stamp}-{hashlib.sha256(f'{ds.sha256}{now}{cfg.target_column}'.encode()).hexdigest()[:6]}"
    series_ok = [s for s in (response or {}).get("series") or [] if isinstance(s, dict) and s.get("status") == "ok"]
    tails = {}
    for s in series_ok:
        try:
            h = VM.history_frame(ds.frame, cfg.timestamp_column, s.get("target"), cfg.group_column, s.get("group")).dropna(subset=["value"]).tail(TAIL_POINTS)
            tails[str(s.get("group"))] = [[t.isoformat(), float(v)] for t, v in zip(h["timestamp"], h["value"])]
        except Exception:                                         # noqa: BLE001  (a chart convenience; the record is still valid without it)
            tails[str(s.get("group"))] = []
    return {"schema": RECORD_SCHEMA, "id": rid, "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "created_epoch": now, "mode": mode,
            "status": "ok" if series_ok else "failed", "error": error,
            "dataset": {"name": ds.name, "sha256": ds.sha256, "rows": int(len(ds.frame)), "columns": int(ds.frame.shape[1]), "is_sample": bool(ds.is_sample)},
            "config": {"timestamp_column": cfg.timestamp_column, "target_column": cfg.target_column, "group_column": cfg.group_column, "frequency": cfg.frequency,
                       "horizon": cfg.horizon, "allowed_models": cfg.allowed_models},
            "seconds": seconds, "response": response, "explanation": explanation if isinstance(explanation, dict) and not explanation.get("_unavailable") else None,
            "observed_tail": tails, "tail_points": TAIL_POINTS, "progress": progress}


def valid_record(r) -> bool:
    return (isinstance(r, dict) and r.get("schema") == RECORD_SCHEMA and isinstance(r.get("id"), str) and isinstance(r.get("dataset"), dict)
            and isinstance(r.get("config"), dict) and r.get("status") in ("ok", "failed") and (r.get("status") == "failed" or not VM.check_forecast_response(r.get("response"))))


class SessionStore:
    kind, location = "session", None

    def __init__(self, bucket: list | None = None):
        self.bucket = bucket if bucket is not None else []

    def add(self, record: dict) -> str:
        self.bucket.append(json.loads(json.dumps(record)))
        return record["id"]

    def all(self) -> list[dict]:
        return sorted((r for r in self.bucket if valid_record(r)), key=lambda r: r.get("created_epoch", 0), reverse=True)

    def get(self, rid: str) -> dict | None:
        return next((r for r in self.all() if r["id"] == rid), None)

    def delete(self, rid: str) -> bool:
        n = len(self.bucket)
        self.bucket[:] = [r for r in self.bucket if r.get("id") != rid]
        return len(self.bucket) < n

    def skipped(self) -> int:
        return 0


class FolderStore:
    """One file per run: <folder>/run_<id>.json. The id comes from the clock and a hash, and a file is opened with 'x'
    (create only), so an existing record can never be overwritten."""
    kind = "folder"

    def __init__(self, folder):
        self.folder = Path(folder)
        self.location = str(self.folder)
        self._skipped = 0

    def _path(self, rid: str) -> Path:
        return self.folder / f"run_{slug(rid, 60)}.json"

    def add(self, record: dict) -> str:
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self._path(record["id"]), "x", encoding="utf-8") as f:
            json.dump(record, f, allow_nan=False)
        return record["id"]

    def all(self) -> list[dict]:
        out, self._skipped = [], 0
        if not self.folder.is_dir():
            return out
        for p in sorted(self.folder.glob("run_*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                r = None
            if valid_record(r):
                out.append(r)
            else:
                self._skipped += 1                                # unreadable or altered files are not shown, and are counted
        return sorted(out, key=lambda r: r.get("created_epoch", 0), reverse=True)

    def get(self, rid: str) -> dict | None:
        return next((r for r in self.all() if r["id"] == rid), None)

    def delete(self, rid: str) -> bool:
        p = self._path(rid)
        if p.is_file() and p.parent == self.folder:
            p.unlink()
            return True
        return False

    def skipped(self) -> int:
        return self._skipped


def tail_frame(record: dict, group) -> pd.DataFrame:
    rows = (record.get("observed_tail") or {}).get(str(group)) or []
    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["kind"] = "Observed"
    return df
