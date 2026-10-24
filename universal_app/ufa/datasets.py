"""Upload parsing and dataset inspection for the interface. Pure functions (no Streamlit), so they are unit-testable.

Nothing here changes the data: no filling, dropping, resampling or transforming. Everything reported is labelled
"inferred" and the user must confirm the timestamp and target columns explicitly.

Parsing contract (Phase 7)
    * Input: UTF-8 text (an optional UTF-8 byte-order mark is ignored), comma-separated, one header row, every row
      with exactly as many fields as the header. Anything else is refused with a reason; nothing is repaired.
    * A decimal number becomes the IEEE-754 double nearest to its text (float_precision="round_trip"), so
      float(text) == value for every cell. A whole number that fits in 64 bits stays an exact integer.
    * The tokens pandas treats as missing by default (empty field, NA, N/A, NaN, null, None, ...) are read as missing
      and counted; they are never filled. "inf" / "Infinity" are read as infinite and reported, not hidden.
    * Completely blank lines are ignored (they hold no data). Text is kept as text.
    * The bytes are never written to disk and the file name is used for display only.
"""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---- upload limits: THE single source for the interface (the form, the messages and the user guide quote these).
# Measured in Phase 7 as workable on a 24 GB laptop for one user; do not raise without re-measuring.
# universal_app/launch.py passes MAX_UPLOAD_MB to Streamlit as a literal option; a test keeps the two equal.
SUPPORTED_EXTENSIONS = (".csv",)
# The hosted copy (started through streamlit_app.py, which sets UFA_HOSTED=1) runs on a small shared server
# (690 MB to 2.7 GB of memory), so it accepts smaller files: Phase 7 measured about 400 MB peak at 500,000 rows.
HOSTED = os.environ.get("UFA_HOSTED") == "1"
MAX_UPLOAD_MB = 15 if HOSTED else 50
MAX_ROWS = 250_000 if HOSTED else 2_000_000
MAX_COLUMNS = 200
PREVIEW_ROWS = 10
MAX_FIELD_CHARS = 10_000          # longest single cell accepted
MAX_HEADER_CHARS = 200            # longest column name accepted
MAX_NAME_CHARS = 120              # file name shown in the interface
MAX_EXACT_INT = 2 ** 53           # largest whole number a double holds exactly (the engine computes in doubles)
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_BOMS = ((codecs.BOM_UTF32_LE, "UTF-32"), (codecs.BOM_UTF32_BE, "UTF-32"), (codecs.BOM_UTF16_LE, "UTF-16"), (codecs.BOM_UTF16_BE, "UTF-16"))


class UploadError(ValueError):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    size_bytes: int
    sha256: str
    is_sample: bool = False


@dataclass
class ColumnInfo:
    name: str
    inferred_type: str            # numeric | datetime-like | text | boolean | empty
    missing: int
    distinct: int
    timestamp_candidate: bool = False
    target_candidate: bool = False


@dataclass
class Inspection:
    rows: int
    columns: int
    column_info: list = field(default_factory=list)
    timestamp_candidates: list = field(default_factory=list)
    target_candidates: list = field(default_factory=list)
    duplicate_rows: int = 0


def display_name(name) -> str:
    """The uploaded file's name for DISPLAY only: no directory part, no control characters, bounded length.
    The interface never opens, creates or writes a path derived from it."""
    base = re.split(r"[\\/]", str(name or ""))[-1]
    base = _CTRL.sub("", base).strip()
    if len(base) > MAX_NAME_CHARS:
        base = base[:MAX_NAME_CHARS - 5] + "…" + base[-4:]
    return base or "upload.csv"


def _check_utf8(data: bytes) -> None:
    """Validate the encoding in 1 MB pieces, without building a second full copy of the file as text."""
    dec = codecs.getincrementaldecoder("utf-8")()
    view = memoryview(data)
    try:
        for i in range(0, len(view), 1 << 20):
            dec.decode(view[i:i + (1 << 20)])
        dec.decode(b"", final=True)
    except UnicodeDecodeError:
        raise UploadError("NOT_TEXT", "The file is not UTF-8 text.", "Export the data again as a UTF-8 CSV.") from None


def _scan_structure(data: bytes) -> tuple[list[str], int]:
    """One streaming pass with Python's csv reader: header names and the number of data rows. Refuses rows whose field
    count differs from the header (pandas would silently pad short rows with missing values) and over-long fields."""
    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="")
    reader = csv.reader(text)
    header, n = None, 0
    try:
        for row in reader:
            if not row:                                   # a completely blank line carries no data
                continue
            if header is None:
                header = row
                if len(header) > MAX_COLUMNS:
                    raise UploadError("TOO_LARGE", f"The file has {len(header):,} columns; the limit is {MAX_COLUMNS}.", "Upload an extract with fewer columns.")
                if len(header) < 2:
                    raise UploadError("TOO_FEW_COLUMNS", "The file has fewer than two columns.",
                                      "A forecast needs a timestamp column and at least one numeric column; check the delimiter is a comma (not a semicolon or a tab).")
                continue
            n += 1
            if len(row) != len(header):
                raise UploadError("INCONSISTENT_COLUMNS", f"Data row {n:,} has {len(row)} fields but the header has {len(header)}.",
                                  "Every row must have the same number of fields as the header. Fix the export (check for stray commas or unquoted text); "
                                  "rows are not padded or trimmed here.")
            if n > MAX_ROWS:
                raise UploadError("TOO_LARGE", f"The file has more than {MAX_ROWS:,} data rows; that is the limit.", "Upload a smaller extract.")
            for cell in row:
                if len(cell) > MAX_FIELD_CHARS:
                    raise UploadError("FIELD_TOO_LONG", f"Data row {n:,} has a field of {len(cell):,} characters; the limit is {MAX_FIELD_CHARS:,}.",
                                      "Remove very long free-text fields, or check for an unclosed quote.")
    except csv.Error as ex:
        if "field larger than field limit" in str(ex):
            raise UploadError("FIELD_TOO_LONG", f"A field near data row {n + 1:,} is longer than the limit of {MAX_FIELD_CHARS:,} characters.",
                              "Remove very long free-text fields, or check for an unclosed quote.") from None
        raise UploadError("MALFORMED_CSV", "The file could not be read as CSV.", "Check the delimiter and the quoting.") from None
    if header is None:
        raise UploadError("EMPTY_FILE", "The file has no columns.", "Upload a CSV with a header row and at least one data row.")
    return header, n


def parse_upload(name: str, data: bytes, is_sample: bool = False) -> Dataset:
    """CSV bytes -> Dataset, or UploadError with an actionable message. Values are read as-is (see the module contract)."""
    name = display_name(name)
    if not name.lower().endswith(SUPPORTED_EXTENSIONS):
        raise UploadError("UNSUPPORTED_FORMAT", f"'{name}' is not a supported file type.", "Upload a .csv file (comma-separated, one header row).")
    if len(data) == 0:
        raise UploadError("EMPTY_FILE", "The file is empty.", "Upload a CSV with a header row and at least one data row.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError("FILE_TOO_LARGE", f"The file is {len(data) / 1048576:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.", "Upload a smaller extract.")
    for bom, enc in _BOMS:
        if data.startswith(bom):
            raise UploadError("NOT_TEXT", f"The file is {enc} text, not UTF-8.", "Export the data again as a UTF-8 CSV.")
    if b"\x00" in data:
        raise UploadError("NOT_TEXT", "The file contains NUL bytes, so it is not plain UTF-8 text.", "Export the data again as a UTF-8 CSV.")
    _check_utf8(data)
    header, n_rows = _scan_structure(data)
    if any(not h.strip() for h in header):
        raise UploadError("BLANK_COLUMN_NAME", "At least one column has no name in the header row.", "Give every column a name; names are not invented here.")
    if any(len(h) > MAX_HEADER_CHARS for h in header):
        raise UploadError("COLUMN_NAME_TOO_LONG", f"A column name is longer than {MAX_HEADER_CHARS} characters.", "Shorten the column names.")
    if any(_CTRL.search(h) for h in header):
        raise UploadError("INVALID_COLUMN_NAME", "A column name contains a control character (for example a line break or a tab).", "Use plain text column names.")
    if len(set(header)) != len(header):
        raise UploadError("DUPLICATE_COLUMNS", "Two or more columns have the same name.", "Rename the columns so each name is unique.")
    if len(header) < 2:
        raise UploadError("TOO_FEW_COLUMNS", "The file has fewer than two columns.", "A forecast needs a timestamp column and at least one numeric column; check the delimiter is a comma.")
    if n_rows == 0:
        raise UploadError("NO_ROWS", "The file has a header but no data rows.", "Upload a CSV that contains observations.")
    try:
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", float_precision="round_trip")     # each number is the exact nearest value to its text
    except (pd.errors.EmptyDataError,):
        raise UploadError("EMPTY_FILE", "The file has no columns.", "Upload a CSV with a header row and at least one data row.") from None
    except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as ex:
        raise UploadError("MALFORMED_CSV", "The file could not be read as CSV.", f"Check the delimiter, quoting and that every row has the same number of fields ({type(ex).__name__}).") from None
    except MemoryError:
        raise UploadError("TOO_LARGE_FOR_MEMORY", "This computer ran out of memory while reading the file.", "Upload a smaller extract.") from None
    if [str(c) for c in df.columns] != header or len(df) != n_rows:      # two independent readers must agree; otherwise nothing is trusted
        raise UploadError("MALFORMED_CSV", "The file's rows or column names could not be read unambiguously.",
                          "Check for unusual quoting, stray line breaks inside fields, or a header that looks like data.")
    df.columns = header
    return Dataset(name, df, len(data), hashlib.sha256(data).hexdigest(), is_sample)


def _datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or s.dropna().empty:
        return False
    sample = s.dropna().astype(str).head(200)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
    except (ValueError, TypeError):
        return False
    return bool(parsed.notna().mean() >= 0.95)


def inspect(df: pd.DataFrame) -> Inspection:
    out = Inspection(len(df), df.shape[1], duplicate_rows=int(df.duplicated().sum()))
    for c in df.columns:
        s = df[c]
        if s.dropna().empty:
            kind = "empty"
        elif pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif _datetime_like(s):
            kind = "datetime-like"
        else:
            kind = "text"
        info = ColumnInfo(c, kind, int(s.isna().sum()), int(s.nunique(dropna=True)), kind == "datetime-like", kind == "numeric" and s.nunique(dropna=True) > 1)
        out.column_info.append(info)
    out.timestamp_candidates = [i.name for i in out.column_info if i.timestamp_candidate]
    out.target_candidates = [i.name for i in out.column_info if i.target_candidate]
    return out


@dataclass
class TimestampCheck:
    parseable: bool
    unparseable: int = 0
    start: str | None = None
    end: str | None = None
    duplicate_timestamps: int = 0
    sorted: bool = True
    inferred_frequency: str | None = None       # informational only; the engine decides
    blocking: list = field(default_factory=list)   # [(code, message)]: reasons the column cannot be used as it is
    notes: list = field(default_factory=list)


_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2}|\d{4})(?!\d)")
ISO_HINT = "Rewrite the dates year-first, for example 2024-01-31 or 2024-01-31 14:00:00."


def _date_order_problem(text: pd.Series, n_unparsed: int) -> tuple[str, str] | None:
    """Dates written as three number groups with the year last (03/04/2024) do not say which group is the day.
    Returns (code, message) when the engine could misread them; None when the text is unambiguous."""
    m = text.str.extract(_NUMERIC_DATE)
    hit = m[0].notna()
    if hit.mean() < 0.95:
        return None
    a, b, y = m.loc[hit, 0].astype(int), m.loc[hit, 1].astype(int), m.loc[hit, 2]
    if (y.str.len() == 2).any():
        return "TIMESTAMP_TWO_DIGIT_YEAR", "The dates use two-digit years, so the century and the day/month order are not certain. " + ISO_HINT
    if a.max() <= 12 and b.max() <= 12:
        return ("TIMESTAMP_AMBIGUOUS_ORDER", "Every date could be read as day/month or as month/day (for example 03/04/2024), and the two readings give different "
                "timestamps. The app does not guess. " + ISO_HINT)
    if a.max() > 12 and b.max() > 12:
        return "TIMESTAMP_AMBIGUOUS_ORDER", "The dates mix day-first and month-first writing. " + ISO_HINT
    if a.max() > 12 and n_unparsed:
        # Clearly day-first. The reader settles the order from the first row; when that row is itself ambiguous it picks
        # month-first and then cannot read the rows whose day is above 12. (With no unreadable row the order was read correctly.)
        return ("TIMESTAMP_DAY_FIRST", f"The dates are written day-first (for example 31/01/2024) but begin with a date that also reads as month-first, so "
                f"{n_unparsed:,} of them cannot be read reliably. " + ISO_HINT)
    return None                                            # month-first, or day-first read without a single failure


def check_timestamps(df: pd.DataFrame, column: str, group: str | None = None) -> TimestampCheck:
    """Facts about the chosen timestamp column (shown as 'inferred'); nothing is modified. `blocking` lists reasons the
    column cannot be handed to the engine without a risk of the timestamps being misread."""
    raw = df[column]
    if pd.api.types.is_bool_dtype(raw) or pd.api.types.is_numeric_dtype(raw):
        return TimestampCheck(False, blocking=[("TIMESTAMP_NUMERIC", f"'{column}' holds plain numbers, not dates. Numbers such as years (2024), Excel serial dates or Unix "
                                                "times do not say what unit they are in, and would be misread as fractions of a second after 1970. "
                                                "Convert the column to dates. " + ISO_HINT)])
    blocking, notes = [], []
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(raw, errors="coerce")
    except (ValueError, TypeError, OverflowError) as ex:
        why = "mix different time-zone offsets" if "timezone" in str(ex).lower() else "could not be read consistently"
        return TimestampCheck(False, blocking=[("TIMESTAMP_MIXED_TIMEZONES" if "timezone" in str(ex).lower() else "TIMESTAMP_UNREADABLE",
                                                f"The values in '{column}' {why}. Use one time zone (or none) for the whole column. " + ISO_HINT)])
    bad = int(ts.isna().sum())
    if bad == len(ts):
        return TimestampCheck(False, bad)
    if not pd.api.types.is_datetime64_any_dtype(raw):
        prob = _date_order_problem(raw.dropna().astype(str), int((ts.isna() & raw.notna()).sum()))
        if prob:
            blocking.append(prob)
    if getattr(ts.dt, "tz", None) is not None:
        notes.append("The timestamps carry a time-zone offset. The engine converts them to UTC and says so in its warnings.")
    dup = int(pd.DataFrame({"g": df[group] if group else 0, "t": ts}).dropna().duplicated().sum())
    ok = ts.dropna()
    freq = None
    try:
        d = ok.sort_values().diff().dropna()
        if len(d) and not group:
            freq = str(d.mode().iloc[0])
    except (TypeError, ValueError):
        freq = None
    return TimestampCheck(bad == 0, bad, str(ok.min()), str(ok.max()), dup, bool(ok.is_monotonic_increasing) if not group else True, freq, blocking, notes)


def target_problems(df: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    """Reasons a column cannot be forecast exactly as uploaded: [(code, message)]. Nothing is changed."""
    s = df[column]
    if pd.api.types.is_bool_dtype(s):
        return [("TARGET_NOT_NUMERIC", "The target column holds true/false values, not numbers.")]
    if not pd.api.types.is_numeric_dtype(s):
        return [("TARGET_NOT_NUMERIC", "The target column must be numeric. Text such as '1,234.5' (thousands separators), '12,5' (decimal comma) or units is not converted.")]
    out = []
    if pd.api.types.is_float_dtype(s):
        n_inf = int(np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).sum())
        if n_inf:
            out.append(("TARGET_NON_FINITE", f"The target has {n_inf:,} infinite values (written as inf / Infinity, or too large for a number). "
                                             "They are not treated as missing; replace or remove them."))
    elif pd.api.types.is_integer_dtype(s) and len(s) and int(s.abs().max()) > MAX_EXACT_INT:
        out.append(("TARGET_NOT_EXACT", f"The target has whole numbers larger than {MAX_EXACT_INT:,}. The engine computes in double precision, which cannot hold "
                                        "them exactly, so they would be changed. Rescale the values (for example to thousands) first."))
    return out


def group_problems(df: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    n = int(df[column].isna().sum())
    if n:
        return [("GROUP_HAS_MISSING", f"The series identifier column has {n:,} empty values (or values read as missing, such as NA). Those rows would be left out of "
                                      "every series. Give each row an identifier.")]
    return []

# ---------------------------------------------------------------- automatic configuration (suggestions only)
FREQ_BY_DELTA = {pd.Timedelta("1min"): "min", pd.Timedelta("5min"): "5min", pd.Timedelta("15min"): "15min", pd.Timedelta("30min"): "30min", pd.Timedelta("1h"): "h",
                 pd.Timedelta("1D"): "D", pd.Timedelta("7D"): "7D"}
DEFAULT_HORIZON = {"min": 60, "5min": 12, "15min": 96, "30min": 48, "h": 24, "D": 7, "B": 5, "7D": 4, "MS": 12, "QS": 4}
HORIZON_WHY = {"min": "one hour", "5min": "one hour", "15min": "one day", "30min": "one day", "h": "one day", "D": "one week", "B": "one business week", "7D": "four weeks",
               "MS": "one year", "QS": "one year"}
_ID_NAME = re.compile(r"(^|[^a-z])(id|idx|index|row|rownum|no|num|number|code|key|zip|postcode|pin|phone|year|serial|seq|sku)([^a-z]|$)", re.I)


@dataclass
class Suggestion:
    """What the app could work out from the file. Every item is a suggestion the user can change; `reasons` says, in
    plain words, what each one is based on, and `needs_confirmation` lists the items where a wrong guess would change
    the result and more than one answer is plausible."""
    timestamp: str | None = None
    target: str | None = None
    group: str | None = None
    frequency: str | None = None          # pandas alias, or None when it could not be determined (the engine then infers or refuses)
    horizon: int | None = None
    reasons: dict = field(default_factory=dict)
    timestamp_candidates: list = field(default_factory=list)
    suitable_targets: list = field(default_factory=list)
    unsuitable: dict = field(default_factory=dict)        # column -> why it is not offered as a target
    needs_confirmation: list = field(default_factory=list)
    n_series: int = 1


def _target_reason(df: pd.DataFrame, c: str, n_series: int) -> str | None:
    """Why a numeric column should NOT be offered as a forecast target, or None when it is a plausible one."""
    s = df[c]
    if pd.api.types.is_bool_dtype(s):
        return "true/false values"
    if not pd.api.types.is_numeric_dtype(s):
        return "not numeric"
    v = s.dropna()
    if v.empty:
        return "empty"
    if v.nunique() <= 1:
        return "constant"
    if pd.api.types.is_integer_dtype(s):
        d = v.diff().dropna()
        if n_series == 1 and len(d) and (d == d.iloc[0]).all() and d.iloc[0] != 0:
            return "counts up by a fixed step (looks like a row number)"
        idlike = bool(_ID_NAME.search(str(c)))
        if v.nunique() == len(v) and idlike:
            return "every value is different and the name reads like an identifier"
        if v.nunique() <= 12 and (idlike or v.nunique() <= 5):
            return f"only {v.nunique()} different whole numbers (looks like a category code)"
        if idlike and v.nunique() <= max(50, len(v) // 20):
            return "the name reads like an identifier or code"
    return None


def _infer_frequency(ts: pd.Series) -> tuple[str | None, str]:
    t = pd.Series(pd.to_datetime(ts, errors="coerce").dropna().unique()).sort_values()
    if len(t) < 3:
        return None, "too few timestamps to judge the spacing"
    d = t.diff().dropna()
    modal = d.mode().iloc[0]
    share = float((d == modal).mean())
    if (t.dt.day == 1).all() and (t.dt.normalize() == t).all():
        months = (t.dt.year * 12 + t.dt.month).diff().dropna()
        if (months == 1).mean() >= 0.95:
            return "MS", "every timestamp is the first day of a month, one month apart"
        if (months == 3).mean() >= 0.95:
            return "QS", "every timestamp is the first day of a quarter, three months apart"
    if modal == pd.Timedelta("1D") and (t.dt.dayofweek < 5).all() and (t.dt.normalize() == t).all() and share < 0.97 and d.max() <= pd.Timedelta("5D"):
        return "B", "daily dates with no Saturdays or Sundays"
    if share >= 0.8 and modal in FREQ_BY_DELTA:
        unit = {"min": "one minute", "5min": "5 minutes", "15min": "15 minutes", "30min": "30 minutes", "h": "one hour", "D": "one day", "7D": "one week"}[FREQ_BY_DELTA[modal]]
        return FREQ_BY_DELTA[modal], f"{share:.0%} of consecutive timestamps are {unit} apart"
    if share >= 0.8:
        return None, f"the usual gap is {modal}, which is not one of the listed frequencies; the engine will work it out from the timestamps"
    return None, f"the spacing is irregular: the most common gap ({modal}) covers only {share:.0%} of the timestamps"


def suggest_config(df: pd.DataFrame, ins: Inspection | None = None) -> Suggestion:
    """Inspect the actual values (never just the column names) and propose a configuration. Nothing is changed in the data."""
    ins = ins or inspect(df)
    sg = Suggestion()
    # ---- timestamp: a column whose VALUES parse as dates without ambiguity
    valid = {}
    for c in ins.timestamp_candidates:
        tc = check_timestamps(df, c)
        if tc.parseable and not tc.blocking:
            valid[c] = tc
    sg.timestamp_candidates = list(valid)
    if not valid:
        blocked = [c for c in ins.timestamp_candidates if c not in valid]
        sg.reasons["timestamp"] = ("No column could be read as dates without doubt" + (f" ('{blocked[0]}' looks like dates but is ambiguous or incomplete)" if blocked else "")
                                   + ". Choose the timestamp column yourself.")
        sg.needs_confirmation.append("timestamp")
    else:
        sg.timestamp = next(iter(valid))
        if len(valid) == 1:
            sg.reasons["timestamp"] = f"'{sg.timestamp}' is the only column whose values are all valid dates."
        else:
            sg.reasons["timestamp"] = f"{len(valid)} columns hold valid dates ({', '.join(valid)}); the first is suggested."
            sg.needs_confirmation.append("timestamp")
    # ---- series identifier: only when timestamps repeat and one column separates them cleanly
    tc = valid.get(sg.timestamp)
    if tc is not None and tc.duplicate_timestamps:
        ts = pd.to_datetime(df[sg.timestamp], errors="coerce")
        options = []
        for c in df.columns:
            if c == sg.timestamp or df[c].isna().any():
                continue
            k = int(df[c].nunique())
            if 2 <= k <= max(2, min(1000, len(df) // 3)) and not pd.DataFrame({"g": df[c], "t": ts}).duplicated().any():
                options.append((not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])), k, c))
        if options:
            options.sort()
            sg.group, sg.n_series = options[0][2], options[0][1]
            sg.reasons["group"] = f"Timestamps repeat, and '{sg.group}' splits the rows into {sg.n_series} series with no repeated timestamp inside any of them."
            if len(options) > 1:
                sg.needs_confirmation.append("group")
        else:
            sg.reasons["group"] = "Timestamps repeat and no column separates the rows into clean series. Remove the duplicates or add a series identifier."
            sg.needs_confirmation.append("group")
    else:
        sg.reasons["group"] = "Each timestamp appears once, so the file is treated as one series."
    # ---- target: numeric columns that are not identifiers, codes, constants or the columns already used
    for c in df.columns:
        if c in (sg.timestamp, sg.group) or c in sg.timestamp_candidates:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        why = _target_reason(df, c, sg.n_series)
        if why:
            sg.unsuitable[c] = why
        else:
            sg.suitable_targets.append(c)
    if not sg.suitable_targets:
        sg.reasons["target"] = "No numeric column looks like a quantity to forecast. Choose the target yourself."
        sg.needs_confirmation.append("target")
    else:
        sg.target = sg.suitable_targets[0]
        if len(sg.suitable_targets) == 1:
            sg.reasons["target"] = f"'{sg.target}' is the only numeric column that looks like a measured quantity."
        else:
            sg.reasons["target"] = (f"{len(sg.suitable_targets)} numeric columns could be forecast ({', '.join(sg.suitable_targets[:6])}); "
                                    f"'{sg.target}' is first in the file. Only you know which one you want.")
            sg.needs_confirmation.append("target")
    # ---- frequency and a default horizon
    if sg.timestamp:
        one = df if not sg.group else df[df[sg.group] == df[sg.group].iloc[0]]
        sg.frequency, why = _infer_frequency(one[sg.timestamp])
        sg.reasons["frequency"] = why[0].upper() + why[1:] + "."
        n = len(one)
        base = DEFAULT_HORIZON.get(sg.frequency)
        cap = max(1, min((n - 1) // 5, 1000))
        if base:
            sg.horizon = max(1, min(base, cap))
            sg.reasons["horizon"] = (f"{sg.horizon} steps: {HORIZON_WHY[sg.frequency]} ahead at this frequency." if sg.horizon == base
                                     else f"{sg.horizon} steps: shortened from {base} because the history is short ({n} rows).")
        else:
            sg.horizon = max(1, min(10, cap))
            sg.reasons["horizon"] = f"{sg.horizon} steps: a short default, because the frequency could not be named."
    return sg
