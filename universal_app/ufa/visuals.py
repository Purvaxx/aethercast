"""HTML for the data-driven visuals on the Upload, Dataset Preview and Configure Forecast pages.

Everything is computed from the uploaded file as read (no filling, resampling or smoothing). Values are only scaled to
pixels. Every piece of text from the file goes through VM.safe() before it is placed in markup, and the snippets
contain no script: motion comes from the shared CSS in components.py (it switches off under "reduce motion").
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import viewmodels as VM

ROLE_LABEL = {"timestamp": "time", "target": "target", "group": "series", "numeric": "numeric", "text": "text", "boolean": "yes/no", "empty": "empty",
              "datetime-like": "dates", "id": "identifier"}


def column_roles(ins, sg) -> dict:
    """One short role per column, from the inspection and the suggestion (both already computed from the file)."""
    out = {}
    for c in ins.column_info:
        if c.name == sg.timestamp:
            out[c.name] = "timestamp"
        elif c.name == sg.target:
            out[c.name] = "target"
        elif c.name == sg.group:
            out[c.name] = "group"
        elif c.name in sg.unsuitable and c.inferred_type == "numeric":
            out[c.name] = "id"
        else:
            out[c.name] = c.inferred_type
    return out


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or (not isinstance(v, (str, bytes)) and pd.isna(v)):
        return "<td class='miss' title='missing value'>missing</td>"
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return f"<td class='n'>{VM.safe(repr(v.item() if hasattr(v, 'item') else v))}</td>"
    return f"<td>{VM.safe(str(v), 80)}</td>"


def preview_table(df: pd.DataFrame, roles: dict, rows: int, max_cols: int = 40) -> str:
    """The first rows exactly as read. Rows fade in one after another; hovering a row or a column highlights it."""
    cols = list(df.columns)[:max_cols]
    head = "".join(f"<th><span>{VM.safe(c, 60)}</span><em class='r-{VM.safe(roles.get(c, ''))}'>{VM.safe(ROLE_LABEL.get(roles.get(c, ''), roles.get(c, '')))}</em></th>" for c in cols)
    body = "".join(f"<tr style='--i:{i}'><td class='ix'>{i + 1}</td>" + "".join(_cell(v) for v in row) + "</tr>"
                   for i, row in enumerate(df[cols].head(rows).itertuples(index=False, name=None)))
    n = len(cols) + 1
    hover = "".join(f".ufx-ptable table:has(td:nth-child({k}):hover) :is(td,th):nth-child({k}){{background:rgba(255,255,255,.055)}}" for k in range(2, n + 1))
    more = f"<div class='ufx-basis'>Showing the first {max_cols} of {len(df.columns)} columns here; all columns are in the spreadsheet view below.</div>" if len(df.columns) > max_cols else ""
    return (f"<style>{hover}</style><div class='ufx-ptable ufx-enter'><div class='sc'><table><thead><tr><th class='ix'>#</th>{head}</tr></thead><tbody>{body}</tbody></table></div></div>{more}")


def _spark(vals: list[float], w: float = 160.0, h: float = 34.0) -> str:
    if len(vals) < 2:
        return "<svg class='ufx-cspark' viewBox='0 0 160 34'></svg>"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    xy = [(w * i / (len(vals) - 1), 3 + (h - 6) * (1 - (v - lo) / rng)) for i, v in enumerate(vals)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    length = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(xy, xy[1:]))
    return f"<svg class='ufx-cspark' viewBox='0 0 {w:.0f} {h:.0f}' preserveAspectRatio='none' aria-hidden='true' style='--len:{length + 4:.0f}'><path d='{d}'/></svg>"


def column_profiles(df: pd.DataFrame, ins, roles: dict, max_cols: int = 40) -> str:
    """One row per column: role, type, a sparkline of its values in file order (numeric columns), missing share and distinct count."""
    rows = []
    n = max(len(df), 1)
    for i, c in enumerate(ins.column_info[:max_cols]):
        role = roles.get(c.name, c.inferred_type)
        spark = ""
        if c.inferred_type == "numeric":
            s = pd.to_numeric(df[c.name], errors="coerce")
            s = s[np.isfinite(s)] if len(s) else s
            if len(s) > 1:
                step = max(1, len(s) // 80)
                spark = _spark([float(v) for v in s.iloc[::step].tolist()])
        share = c.missing / n
        rows.append(f"<div class='ufx-prof' style='--i:{i}'><div class='nm'><b>{VM.safe(c.name, 60)}</b><em class='r-{VM.safe(role)}'>{VM.safe(ROLE_LABEL.get(role, role))}</em></div>"
                    f"<div class='sp'>{spark or '<span class=\"na\">' + VM.safe(c.inferred_type) + '</span>'}</div>"
                    f"<div class='ms'><span class='bar'><i style='--w:{min(share * 100, 100):.2f}%'></i></span><span>{'none missing' if not c.missing else f'{c.missing:,} missing ({share:.1%})'}</span></div>"
                    f"<div class='ds'>{c.distinct:,} distinct</div></div>")
    return "<div class='ufx-profiles'>" + "".join(rows) + "</div>"


def missing_map(df: pd.DataFrame, bins: int = 48, max_cols: int = 40) -> str:
    """Where values are missing, down the file: each cell is one slice of rows, shaded by the share of missing values in
    that slice (black = none, white = all). Shown only when something is missing."""
    cols = list(df.columns)[:max_cols]
    m = df[cols].isna()
    if not m.values.any():
        return ""
    k = min(bins, len(df))
    edges = np.linspace(0, len(df), k + 1).astype(int)
    lines = []
    for c in cols:
        col = m[c].to_numpy()
        cells = "".join(f"<i style='--a:{col[a:b].mean():.3f}' title='rows {a + 1:,} to {b:,}: {int(col[a:b].sum()):,} missing'></i>" for a, b in zip(edges, edges[1:]) if b > a)
        lines.append(f"<div class='mr'><span>{VM.safe(c, 40)}</span><div class='cells' style='--k:{k}'>{cells}</div></div>")
    return ("<div class='ufx-mmap ufx-enter'>" + "".join(lines) + f"<div class='ufx-basis'>Each square is a slice of about {max(len(df) // k, 1):,} rows, top of the file on the left. "
            "Brighter means more missing values in that slice. Nothing has been filled.</div></div>")


def hero_visual() -> str:
    """Decorative illustration on the Upload page (not data): a line that draws, then a dashed continuation and a band."""
    return ("<svg class='ufx-hero-viz' viewBox='0 0 640 150' aria-hidden='true' preserveAspectRatio='none'>"
            "<defs><linearGradient id='hvg' x1='0' x2='1'><stop offset='0' stop-color='#fff' stop-opacity='0'/><stop offset='.25' stop-color='#fff' stop-opacity='.9'/>"
            "<stop offset='1' stop-color='#fff' stop-opacity='1'/></linearGradient></defs>"
            "<g class='gr'><line x1='0' x2='640' y1='30' y2='30'/><line x1='0' x2='640' y1='75' y2='75'/><line x1='0' x2='640' y1='120' y2='120'/></g>"
            "<path class='band' d='M430,70 C470,52 520,40 560,46 S620,30 640,26 L640,92 C610,96 590,104 560,100 S470,98 430,82 Z'/>"
            "<path class='obs' d='M0,108 C30,100 45,112 70,96 S110,84 135,92 S175,70 200,78 S245,96 270,74 S320,60 345,70 S395,82 430,76'/>"
            "<path class='fc' d='M430,76 C470,62 520,58 560,72 S620,58 640,56'/><line class='split' x1='430' x2='430' y1='14' y2='136'/><circle class='dot' cx='430' cy='76' r='4'/></svg>")
