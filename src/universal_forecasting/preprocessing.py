"""Train-only preprocessing. Parameters are fitted ONCE on rows of the training block and then applied unchanged to
validation and test rows; a second `fit` raises (no refitting on later data)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


class RefitError(RuntimeError):
    pass


@dataclass
class TrainOnlyScaler:
    kind: str = "standard"                  # standard (mean / population std) | robust (median / IQR)
    columns: list = field(default_factory=list)
    center: list = field(default_factory=list)
    scale: list = field(default_factory=list)
    fit_rows: int = 0
    fit_rule: str = ""
    fitted: bool = False

    def fit(self, X: np.ndarray, train_mask: np.ndarray, columns: list[str], fit_rule: str) -> "TrainOnlyScaler":
        if self.fitted:
            raise RefitError("scaler already fitted on the training block; refitting is forbidden")
        x = np.asarray(X, dtype="float64")[np.asarray(train_mask, bool)]
        if not len(x):
            raise ValueError("no training rows to fit on")
        if self.kind == "standard":
            c, s = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        elif self.kind == "robust":
            c = np.nanmedian(x, axis=0)
            s = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        else:
            raise ValueError(self.kind)
        s = np.where(~np.isfinite(s) | (s <= 0), 1.0, s)       # constant / all-NaN in train -> scale 1 (recorded via fit stats)
        c = np.where(np.isfinite(c), c, 0.0)
        self.columns, self.center, self.scale = list(columns), c.tolist(), s.tolist()
        self.fit_rows, self.fit_rule, self.fitted = int(len(x)), fit_rule, True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit on the training block first")
        return ((np.asarray(X, dtype="float64") - np.asarray(self.center)) / np.asarray(self.scale)).astype("float32")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainOnlyScaler":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
