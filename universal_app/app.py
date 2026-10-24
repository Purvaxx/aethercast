"""AetherCast AI: Streamlit interface.

    python universal_app/launch.py            (port 8503; normal mode: the forecasting engine and the rule-based explanations)
    python universal_app/launch.py --demo     (demo mode: stored result for the bundled synthetic sample)

Six pages: Upload, Dataset Preview, Configure Forecast, Forecast Results, Insights, Forecast Reports.
The interface talks to the engine only through ufa/services.py; it never computes, edits or invents a forecast number,
and it makes no language-model call. A result is shown only for the exact file and settings it was computed for.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ufa import components as UI  # noqa: E402
from ufa import datasets as DS  # noqa: E402
from ufa import history as H  # noqa: E402
from ufa import reports as R  # noqa: E402
from ufa import services as SV  # noqa: E402
from ufa import viewmodels as VM  # noqa: E402
from ufa import visuals as VIS  # noqa: E402
from ufa import charts as CH  # noqa: E402

st.set_page_config(page_title="AetherCast AI", page_icon="🪐", layout="wide", initial_sidebar_state="collapsed")
UI.saturn()                                                        # first element on every page, so the background persists across pages
UI.theme()
S = st.session_state
S.setdefault("step", "upload")
try:
    service = S.get("_service") or SV.make_service(os.environ.get(SV.SERVICE_ENV))   # tests inject a MockService via session state
except ValueError as _ex:                                                            # an unknown mode is an error, never a silent default
    st.error(f"Configuration error: {_ex}")
    st.stop()
RESULT_KEYS = ("response", "result_key", "result_meta", "explanation", "qa", "run_error", "run_progress", "current_run_id", "view_record")
WIDGET_RESULT_KEYS = ("txt_q", "pill_q", "sel_series")
PREVIEW_TABLE_ROWS = 25
PAGES = [("upload", "Upload"), ("preview", "Dataset Preview"), ("configure", "Configure Forecast"), ("results", "Forecast Results"), ("insights", "Insights"), ("reports", "Forecast Reports")]
SUGGESTED = ["What does this forecast mean?", "Why was this model selected?", "How reliable is this forecast?", "What do the prediction intervals mean?",
             "Which forecast horizon appears more difficult?", "What limitations should I consider?", "How does this model compare with the baseline?"]
# Each completed stage stays readable for a moment in a real browser session. The pause happens AFTER the stage's real
# event, so the display can lag the work slightly but never runs ahead of it. The launcher sets it; tests use none.
try:
    STAGE_MIN_VISIBLE_S = min(max(float(os.environ.get("UFA_STAGE_PAUSE", "0")), 0.0), 2.0)
except ValueError:
    STAGE_MIN_VISIBLE_S = 0.0


def store():
    """Where completed runs are kept: a local folder when the launcher configured one, otherwise this browser session only."""
    folder = None if DS.HOSTED else os.environ.get(H.STORE_ENV)    # hosted: never a shared folder, so visitors never see each other's runs
    return H.FolderStore(folder) if folder else H.SessionStore(S.setdefault("_runs", []))


HOSTED_HISTORY_NOTE = ("On this hosted version, saved runs last only for this browser session: they are gone when you close or reload the tab, "
                       "after a long pause, or when the app restarts or sleeps. Download the CSV and HTML files to keep a run.")


def go(step: str):
    S.step = step
    st.rerun()


def cached(kind: str, ds: DS.Dataset, key, compute):
    """Inspection results for the loaded file, computed once per file (they scan every row) instead of on every interaction."""
    box = S.get("_cache")
    if not box or box.get("sha") != ds.sha256:
        box = S["_cache"] = {"sha": ds.sha256}
    if (kind, key) not in box:
        box[(kind, key)] = compute()
    return box[(kind, key)]


def withdraw_confirmation():
    """Runs when any forecast setting changes: a confirmation tick is removed (on the page too), so it never vouches for settings the user has not re-read."""
    S["chk_confirm"] = False


def touch(field: str):
    def _cb():
        S.setdefault("touched", set()).add(field)
        withdraw_confirmation()
    return _cb


def clear_result():
    """Forget the last forecast, its explanation and its questions (new dataset, changed settings or a new run). Saved reports are not affected."""
    for k in RESULT_KEYS + WIDGET_RESULT_KEYS:
        S.pop(k, None)


def reset_all():
    clear_result()
    for k in ("dataset", "cfg", "suggestion", "touched", "validation_issues", "upload_error", "_cache", "pending_run", "chk_confirm"):
        S.pop(k, None)
    S["uploader_key"] = S.get("uploader_key", 0) + 1
    go("upload")


def load_dataset(ds: DS.Dataset):
    """A new file: inspect it, work out a suggested configuration from its actual values, and start on the preview page."""
    S.dataset = ds
    S.pop("upload_error", None)
    clear_result()
    for k in ("cfg", "validation_issues", "touched", "chk_confirm", "pending_run"):
        S.pop(k, None)
    ins = cached("inspect", ds, None, lambda: DS.inspect(ds.frame))
    sg = S.suggestion = DS.suggest_config(ds.frame, ins)
    S.cfg = SV.ForecastConfig(sg.timestamp, sg.target, sg.horizon, sg.group, sg.frequency, None, confirmed=not sg.needs_confirmation)
    go("preview")


def start_run():
    S.pop("validation_issues", None)
    S["run_seq"] = S.get("run_seq", 0) + 1
    S["pending_run"] = S["run_seq"]                               # exactly one run is owed for this click
    go("processing")


if S.step not in [p for p, _ in PAGES] + ["processing"]:
    S.step = "upload"
UI.banner(getattr(service, "banner", None))


NAV_ICONS = {"upload": ":material/upload_file:", "preview": ":material/table_view:", "configure": ":material/tune:", "results": ":material/show_chart:",
             "insights": ":material/insights:", "reports": ":material/description:"}


def navigation():
    """The page bar: six real buttons over a sliding indicator. A page is offered only when it has something to show;
    while a forecast runs the bar is locked."""
    have_ds = S.get("dataset") is not None
    have_res = any(k in S for k in ("response", "run_error", "view_record")) or S.step == "processing"
    enabled = {"upload": True, "preview": have_ds, "configure": have_ds, "results": have_res, "insights": True, "reports": True}
    here = "results" if S.step == "processing" else S.step
    bar = st.container(key="ufa_nav")
    with bar:
        UI.nav_state([p for p, _ in PAGES].index(here))            # moves the indicator; the CSS animates the slide
    for col, (page, label) in zip(bar.columns(len(PAGES)), PAGES):
        if col.button(label, key=f"nav_{page}", icon=NAV_ICONS[page], type="primary" if page == here else "secondary",
                      disabled=not enabled[page] or S.step == "processing", width="stretch", help=None):
            if page != S.step:
                go(page)
    UI.saturn_place(S.step)


st.markdown("<div class='ufx-brand' role='banner'><i aria-hidden='true'></i>AetherCast AI</div>", unsafe_allow_html=True)
navigation()


# ====================================================================== 1. upload
CHECK_SVG = "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M5 12.5l4.2 4.2L19 7'/></svg>"


def file_state_html(ds: DS.Dataset) -> str:
    """The loaded file, from its parsed contents: name, rows, columns, size and what the automatic check found."""
    sg = S.get("suggestion")
    checks = []
    if sg is not None:
        checks.append(("ufx-ok", f"time column: {sg.timestamp}") if sg.timestamp else ("ufx-need", "time column: needs your choice"))
        checks.append(("ufx-ok", f"target: {sg.target}") if sg.target else ("ufx-need", "target: needs your choice"))
        checks.append(("ufx-ok" if not sg.needs_confirmation else "ufx-need", "ready to configure" if not sg.needs_confirmation else "confirm on Configure Forecast"))
    st_html = "".join(f"<span class='{c}'>{VM.safe(t, 60)}</span>" for c, t in checks)
    return (f"<div class='ufx-file' role='status'><div class='ic'>{CHECK_SVG}</div><div><div class='nm'>{VM.safe(ds.name)}</div>"
            f"<div class='mt'>{len(ds.frame):,} rows · {ds.frame.shape[1]:,} columns · {ds.size_bytes / 1024:,.1f} KB{' · synthetic sample' if ds.is_sample else ''}</div></div>"
            f"<div class='st'>{st_html}</div></div>")


def screen_upload():
    _, mid, _ = st.columns([0.6, 2.4, 0.6])
    with mid:
        st.markdown("<div class='ufx-hero'><div class='ufx-eyebrow'>AetherCast AI · Upload a dataset</div><h1>Forecast any time series<span>from one CSV file.</span></h1>"
                    "<p>Drop in a file with dates and numbers. It is checked, the settings are worked out from its values, and the engine compares models "
                    "on your own history before it forecasts.</p>" + VIS.hero_visual() + "</div>", unsafe_allow_html=True)
        with st.container(key="ufa_drop"):
            up = st.file_uploader("Drag and drop a CSV file here, or browse", type=["csv"], accept_multiple_files=False, key=f"uploader_{S.get('uploader_key', 0)}",
                                  help="One header row; one column with timestamps; at least one numeric column to forecast.", label_visibility="collapsed")
        if up is not None:
            token = (getattr(up, "file_id", None), up.name, getattr(up, "size", None))
            if S.get("upload_token") != token:                    # act once per chosen file, so coming back to this page does not restart the workflow
                S["upload_token"] = token
                try:
                    load_dataset(DS.parse_upload(up.name, up.getvalue()))
                except DS.UploadError as ex:
                    S.upload_error = {"code": ex.code, "message": ex.message, "hint": ex.hint}
        st.markdown(f"<div class='ufx-chips'><span>CSV · UTF-8 · one header row</span><span>up to {DS.MAX_UPLOAD_MB} MB</span><span>{DS.MAX_ROWS:,} rows</span>"
                    f"<span>{DS.MAX_COLUMNS} columns</span><span>{'kept in memory for your session' if DS.HOSTED else 'stays on this computer'}</span></div>", unsafe_allow_html=True)
        if S.get("upload_error"):
            e = S.upload_error
            with st.container(key="ufa_upload_err"):
                st.error(f"{e['code']}: {e['message']} {e['hint']}", icon=":material/block:")
                st.caption("File not accepted. Fix what is named above and drop the corrected file in the box; nothing from the rejected file was kept.")
        ds = S.get("dataset")
        if ds is not None:
            st.markdown(file_state_html(ds), unsafe_allow_html=True)
            st.caption(f"Current file: {ds.name} ({len(ds.frame):,} rows). Choose another file above to replace it.")
            a, b, _ = st.columns([1.5, 1.2, 1.6])
            if a.button("Open the dataset preview", key="up_open", type="primary", icon=":material/arrow_forward:", width="stretch"):
                go("preview")
            if b.button("Remove file", key="up_remove", icon=":material/close:", width="stretch"):
                reset_all()
        kept = store()
        saved = (f"Completed forecasts are saved on this computer in '{kept.location}' so they appear under Insights and Forecast Reports. The uploaded file itself is not saved."
                 if kept.kind == "folder" else HOSTED_HISTORY_NOTE if DS.HOSTED else "Completed forecasts are kept for this browser session only. The uploaded file is not saved.")
        privacy = ("Your file is sent to this app's server, kept in memory for your session only, never written to disk and never shown to other visitors."
                   if DS.HOSTED else "Nothing is sent off this computer.")
        st.markdown(f"<div class='ufa-center-note'>Limits: {DS.MAX_UPLOAD_MB} MB, {DS.MAX_ROWS:,} rows, {DS.MAX_COLUMNS} columns. {VM.safe(privacy)} {VM.safe(saved)}</div>",
                    unsafe_allow_html=True)
        if SV.SAMPLE_CSV.exists():
            st.markdown("<div class='ufa-section'></div><div class='ufa-center-note'>No file to hand? Try the bundled sample. It is synthetic: computer-generated hourly demand, not real measurements.</div>",
                        unsafe_allow_html=True)
            _, b, _ = st.columns([1, 2, 1])
            if b.button("Load the synthetic sample dataset", key="load_sample", icon=":material/science:", width="stretch"):
                load_dataset(DS.parse_upload(SV.SAMPLE_CSV.name, SV.SAMPLE_CSV.read_bytes(), is_sample=True))
        if service.mode == "not_connected":
            st.info("The forecasting engine is not connected in this mode: you can upload, inspect, configure and validate a request, "
                    "but no forecast will be computed. Nothing is shown in place of a real result.")


def need_dataset() -> DS.Dataset | None:
    ds = S.get("dataset")
    if ds is None:
        UI.empty_state("No dataset loaded", "Upload a CSV to begin.")
        if st.button("Go to upload"):
            go("upload")
    return ds


# ====================================================================== 2. dataset preview
def screen_preview():
    ds = need_dataset()
    if ds is None:
        return
    df = ds.frame
    ins = cached("inspect", ds, None, lambda: DS.inspect(df))
    sg: DS.Suggestion = S.get("suggestion") or cached("suggest", ds, None, lambda: DS.suggest_config(df, ins))
    roles = VIS.column_roles(ins, sg)
    st.markdown(f"<div class='ufx-accept' role='status'>{CHECK_SVG}<b>File accepted</b><span>{VM.safe(ds.name)} · {ins.rows:,} rows · {ins.columns:,} columns · read exactly as written</span></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='ufx-h'><h2>Dataset preview</h2><p>What was read from the file, and what each column appears to be. "
                "Nothing has been changed, filled or reordered.</p></div>", unsafe_allow_html=True)
    if ds.is_sample:
        st.caption("Bundled synthetic sample (computer-generated, not real measurements).")
    k = st.columns(4)
    short = ds.name if len(ds.name) <= 20 else ds.name[:11] + "…" + ds.name[-8:]      # a long name would break mid-word in the tile; the full name is in the line below
    UI.kpi(k[0], "File", short, f"{ds.size_bytes / 1024:,.1f} KB" + ("" if short == ds.name else f" · {ds.name}"))
    UI.kpi(k[1], "Rows", f"{ins.rows:,}")
    UI.kpi(k[2], "Columns", f"{ins.columns:,}")
    tcheck = cached("ts", ds, (sg.timestamp, sg.group), lambda: DS.check_timestamps(df, sg.timestamp, sg.group)) if sg.timestamp else None
    if tcheck and tcheck.start:
        UI.kpi(k[3], "Date range", f"{tcheck.start[:16]} → {tcheck.end[:16]}", f"read from '{sg.timestamp}'")
    else:
        UI.kpi(k[3], "Date range", "not available", "no column could be read as dates without doubt")
    if ins.rows >= SV.LARGE_ROWS and service.mode == "engine":
        st.warning(f"Large file ({ins.rows:,} rows). {SV.LARGE_ROWS_NOTE}")
    if ins.duplicate_rows:
        st.warning(f"{ins.duplicate_rows:,} rows are exact duplicates of another row. They are left in place here; the engine reports how it treats them.")
    if tcheck is not None and tcheck.duplicate_timestamps and not sg.group:
        st.warning(f"Duplicate timestamps: {tcheck.duplicate_timestamps:,} timestamps appear more than once and no column separates them into series.")
    n_missing = sum(c.missing for c in ins.column_info)
    tab_prev, tab_prof, tab_miss, tab_cols = st.tabs(["Preview (first rows)", "Column profiles", f"Missing values ({n_missing:,})", "Columns and data quality"])
    with tab_prev:
        st.markdown(VIS.preview_table(df, roles, PREVIEW_TABLE_ROWS), unsafe_allow_html=True)
        st.caption(f"First {min(PREVIEW_TABLE_ROWS, len(df))} of {len(df):,} rows, exactly as read from the file. Hover a row or a column to follow it; "
                   "hatched cells are missing values. Column labels show the role worked out from the values (time, target, series, numeric, text).")
        with st.expander("Spreadsheet view (sortable, all columns)"):
            st.dataframe(df.head(DS.PREVIEW_ROWS), width="stretch", hide_index=True)
    with tab_prof:
        st.markdown(cached("vis_prof", ds, tuple(sorted(roles.items())), lambda: VIS.column_profiles(df, ins, roles)), unsafe_allow_html=True)
        st.caption("Lines show each numeric column's values in file order (scaled to fit, one point every few rows on long files). Bars show the share of missing values.")
    with tab_miss:
        mm = cached("vis_miss", ds, None, lambda: VIS.missing_map(df)) if n_missing else ""
        if mm:
            st.markdown(mm, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ufx-accept'>{CHECK_SVG}<b>No missing values</b><span>every cell in every column holds a value</span></div>", unsafe_allow_html=True)
    with tab_cols:
        def role(c):
            if c.name == sg.timestamp:
                return "timestamp (suggested)"
            if c.name == sg.group:
                return "series identifier (suggested)"
            if c.name == sg.target:
                return "forecast target (suggested)"
            if c.name in sg.suitable_targets:
                return "possible forecast target"
            if c.name in sg.unsuitable:
                return "not offered as a target: " + sg.unsuitable[c.name]
            return "other dates" if c.name in sg.timestamp_candidates else "not used"
        st.dataframe(pd.DataFrame([{"Column": c.name, "Inferred type": c.inferred_type, "Missing values": c.missing, "Distinct values": c.distinct, "Suggested role": role(c)}
                                   for c in ins.column_info]), width="stretch", hide_index=True)
        st.caption("Types and roles are worked out from the values in each column. They are suggestions; you can change them on the Configure Forecast page.")
    b1, b2, _ = st.columns([2, 1.4, 3])
    if b1.button("Continue to Configure Forecast", type="primary", key="btn_to_config", icon=":material/arrow_forward:", width="stretch"):
        go("configure")
    if b2.button("Remove this file", key="btn_remove_file", icon=":material/close:", width="stretch"):
        reset_all()


# ====================================================================== 3. configure forecast
def hint(field: str, sg: DS.Suggestion, current, suggested):
    """Says where each setting came from: worked out from the file, or changed by the user."""
    touched = field in S.get("touched", set())
    if touched and current != suggested:
        tag, text = UI.badge("Changed by you", "confirmed changed"), ("Suggested value was: " + str(suggested)) if suggested is not None else ""
    elif suggested is not None or (field == "group" and "group" not in sg.needs_confirmation):
        tag, text = UI.badge("Inferred from the file", "inferred"), sg.reasons.get(field, "")
    else:
        tag, text = UI.badge("Needs your input", "confirmed"), sg.reasons.get(field, "")
    st.markdown(f"<div class='ufa-hint'>{tag}{VM.safe(text)}</div>", unsafe_allow_html=True)


def request_summary_html(c: SV.ForecastConfig, sg: DS.Suggestion, tc, freq_label: str, errs: dict, ds: DS.Dataset) -> str:
    """The request in one sentence, rebuilt on every change. Settings changed by the user are highlighted; a short pulse
    marks that the summary was updated. It describes the request only: no forecast value or end date is computed here."""
    sig = (c.timestamp_column, c.target_column, c.group_column, c.horizon, c.frequency, tuple(c.allowed_models or ()))
    if S.get("cfg_sig") != sig:
        S["cfg_rev"] = S.get("cfg_rev", 0) + (1 if "cfg_sig" in S else 0)
        S["cfg_sig"] = sig
    touched = S.get("touched", set())
    def b(text, field):
        return f"<b class='{'chg' if field in touched else ''}'>{VM.safe(text, 60)}</b>"
    if not c.target_column or not c.timestamp_column:
        sent = "Choose " + " and ".join(x for x, ok in (("the timestamp column", c.timestamp_column), ("the column to forecast", c.target_column)) if not ok) + " to complete the request."
        sent = VM.safe(sent)
    else:
        n_series = int(ds.frame[c.group_column].nunique(dropna=True)) if c.group_column and c.group_column in ds.frame.columns else 1
        freq = "the frequency the engine infers from the timestamps" if c.frequency is None else freq_label.lower()
        sent = (f"Forecast {b(c.target_column, 'target')} for {b(str(c.horizon or '?') + ' steps', 'horizon')} ahead, using {b(c.timestamp_column, 'timestamp')} as time, "
                f"at {b(freq, 'frequency')}, " + (f"for each of {b(f'{n_series:,} series', 'group')} in {b(c.group_column, 'group')}." if c.group_column else "as one series.")
                + (f" The history ends at {VM.safe(str(tc.end)[:16])}." if tc is not None and getattr(tc, 'end', None) else ""))
    checks = "".join(f"<span class='{'ufx-need' if k in errs else 'ufx-ok'}'>{VM.safe(lbl)}</span>"
                     for k, lbl in (("timestamp_column", "time column readable"), ("target_column", "target usable"), ("group_column", "series setting valid"), ("horizon", "horizon within limits")))
    pulse = "ufx-pulse-a" if S.get("cfg_rev", 0) % 2 else "ufx-pulse-b"
    return f"<div class='ufx-req {pulse}' role='status' aria-live='polite'><h4>Your request</h4><div class='sent'>{sent}</div><div class='chk'>{checks}</div></div>"


def screen_configure():
    ds = need_dataset()
    if ds is None:
        return
    df = ds.frame
    ins = cached("inspect", ds, None, lambda: DS.inspect(df))
    sg: DS.Suggestion = S.get("suggestion") or cached("suggest", ds, None, lambda: DS.suggest_config(df, ins))
    tcheck_for = lambda col, grp: cached("ts", ds, (col, grp), lambda: DS.check_timestamps(df, col, grp))      # noqa: E731
    cfg: SV.ForecastConfig = S.get("cfg") or SV.ForecastConfig()
    st.markdown("<div class='ufx-h'><h2>Configure forecast</h2><p>Each setting below was worked out from the values in your file. Change anything; the summary updates as you go.</p></div>",
                unsafe_allow_html=True)
    st.caption(f"File: {ds.name}")
    if ins.rows >= SV.LARGE_ROWS and service.mode == "engine":
        st.warning(f"Large file ({ins.rows:,} rows). {SV.LARGE_ROWS_NOTE}")
    cols = list(df.columns)
    ph = {}
    c1, c2 = st.container(key="ufa_cfg").columns(2, gap="large")
    with c1:
        ts = st.selectbox("Timestamp column (required)", cols, index=cols.index(cfg.timestamp_column) if cfg.timestamp_column in cols else None,
                          placeholder="Choose the column that holds the time of each observation", key="sel_ts", on_change=touch("timestamp"),
                          help="Suggested only when a column's values are all valid, unambiguous dates.")
        hint("timestamp", sg, ts, sg.timestamp)
        ph["timestamp_column"] = st.empty()
        tgt = st.selectbox("Target column to forecast (required)", cols, index=cols.index(cfg.target_column) if cfg.target_column in cols else None,
                           placeholder="Choose the numeric column to forecast", key="sel_target", on_change=touch("target"),
                           help="Identifiers, row numbers, codes and constant columns are never suggested.")
        hint("target", sg, tgt, sg.target)
        ph["target_column"] = st.empty()
        grp_opts = ["(none: the file is one series)"] + cols
        grp = st.selectbox("Series identifier column (optional)", grp_opts, index=grp_opts.index(cfg.group_column) if cfg.group_column in cols else 0, key="sel_group",
                           on_change=touch("group"), help="Use this when the file holds several series stacked together (for example one per region).")
        hint("group", sg, None if grp == grp_opts[0] else grp, sg.group)
        ph["group_column"] = st.empty()
    with c2:
        hz = st.number_input("Forecast horizon, in steps (required)", min_value=1, max_value=SV.MAX_HORIZON_UI, value=cfg.horizon, step=1, placeholder="e.g. 24", key="num_h",
                             on_change=touch("horizon"),
                             help="A step is one interval of the series frequency (for hourly data, 24 steps = the next 24 hours). Calendar or business-day horizons are not supported.")
        hint("horizon", sg, int(hz) if hz is not None else None, sg.horizon)
        ph["horizon"] = st.empty()
        flabels = list(SV.FREQUENCY_OPTIONS)
        cur = next((l for l, v in SV.FREQUENCY_OPTIONS.items() if v == cfg.frequency), flabels[0])
        fl = st.selectbox("Frequency of the series", flabels, index=flabels.index(cur), key="sel_freq", on_change=touch("frequency"),
                          help="'Infer from the timestamps' lets the engine work it out. Declare it when the data skips periods by design, e.g. 'Business days'.")
        hint("frequency", sg, SV.FREQUENCY_OPTIONS[fl], sg.frequency)
        with st.expander("Advanced: candidate models"):
            models = st.multiselect("Models the engine may consider", SV.ROUTED_MODELS, default=cfg.allowed_models if cfg.allowed_models is not None else SV.ROUTED_MODELS,
                                    key="ms_models", on_change=touch("models"), help="The naive baseline is always evaluated as the reference and fallback.")
            st.caption("These are the models available in the engine. It compares them on a validation period inside your data and keeps the simpler one when results are effectively tied.")
            ph["allowed_models"] = st.empty()
        summary_box = st.empty()
    new = SV.ForecastConfig(ts, tgt, int(hz) if hz is not None else None, None if grp == grp_opts[0] else grp, SV.FREQUENCY_OPTIONS[fl],
                            None if list(models) == SV.ROUTED_MODELS else list(models))
    # facts about the chosen timestamp column
    tc = None
    if ts:
        tc = tcheck_for(ts, new.group_column)
        if tc.blocking:
            pass                                                  # shown next to the control below, with what to do
        elif not tc.parseable and tc.start is None:
            st.error(f"None of the values in '{ts}' could be read as dates or times. Choose a different timestamp column.")
        else:
            for note in tc.notes:
                st.info(note)
            if tc.unparseable:
                st.warning(f"{tc.unparseable:,} values in '{ts}' could not be read as timestamps. The engine will refuse the dataset until they are fixed or removed.")
            if tc.duplicate_timestamps:
                st.warning(f"Duplicate timestamps: {tc.duplicate_timestamps:,} timestamps appear more than once" + (" within a series" if new.group_column else "")
                           + ". If their values differ, the engine will refuse the dataset; it will not choose between them." +
                           ("" if new.group_column else " If the file holds several series, set the series identifier column."))
            if not tc.sorted:
                st.info("Rows are not in time order. The engine sorts them and reports that it did.")
    if tgt and tgt in df.columns:
        miss = int(df[tgt].isna().sum())
        if miss:
            st.warning(f"The target '{tgt}' has {miss:,} missing values. They are not filled. The engine accepts a small share of missing points and refuses larger gaps.")
        if tgt in sg.unsuitable:
            st.warning(f"'{tgt}' was not suggested as a target because it {sg.unsuitable[tgt]}. You can still forecast it if that is what you want.")
    others = [c for c in cols if c not in (ts, tgt, new.group_column)]
    if others:
        st.caption("The forecast is built from the target's own past values only. Columns that will not be used: " + ", ".join(map(str, others[:8])) + ("…" if len(others) > 8 else "") + ".")
    # ---- confirmation is asked only where the file allows more than one reading and the user has not chosen
    touched = S.get("touched", set())
    current = {"timestamp": (ts, sg.timestamp), "target": (tgt, sg.target), "group": (new.group_column, sg.group)}
    open_items = [f for f in sg.needs_confirmation if f in current and f not in touched and current[f][0] is not None and current[f][0] == current[f][1]]
    if open_items:
        what = " and ".join({"timestamp": f"'{ts}' as the timestamp", "target": f"'{tgt}' as the column to forecast", "group": f"'{new.group_column}' as the series identifier"}[f] for f in open_items)
        label = f"I confirm {what}."
        ticked = st.checkbox(label, key="chk_confirm") if "chk_confirm" in S else st.checkbox(label, value=bool(cfg.confirmed) and (cfg.timestamp_column, cfg.target_column) == (ts, tgt), key="chk_confirm")
        st.caption("Asked because the file allows more than one reading here, and the wrong one would give a forecast of the wrong thing.")
        new.confirmed = bool(ticked)
    else:
        new.confirmed = True
    S.cfg = new
    if S.get("result_key") and S.result_key != SV.request_key(ds.sha256, new):
        clear_result()                                            # the settings changed: the earlier result no longer belongs to this request
    errs = SV.ui_field_errors(df, new, tc)
    for key in ("timestamp_column", "target_column", "group_column", "horizon", "allowed_models"):
        if errs.get(key):
            with ph[key]:
                UI.field_error(errs.get(key))                     # explained next to the setting it concerns
    summary_box.markdown(request_summary_html(new, sg, tc, fl, errs, ds), unsafe_allow_html=True)
    if S.get("validation_issues"):
        st.markdown("**The request was not accepted:**")
        UI.issues_list(S.validation_issues)
    b1, b2, _ = st.columns([2, 1.4, 3])
    if b1.button("Validate and generate forecast", type="primary", disabled=bool(errs), key="btn_run", width="stretch",
                 help="Enabled once the required fields are valid." if errs else None):
        start_run()
    if errs:
        st.caption("To continue: " + " ".join(errs.values()))
    if b2.button("Start over", key="btn_restart_cfg", width="stretch"):
        reset_all()


# ====================================================================== processing (shown under Forecast Results)
def fetch_explanation(resp: dict) -> dict:
    try:
        return service.explain(resp, None, "concise")
    except SV.ServiceUnavailable as ex:
        return {"_unavailable": f"{ex.code}: {ex.message}"}
    except Exception as ex:                                       # noqa: BLE001
        return {"_unavailable": f"EXPLANATION_FAILED: the explanation service failed ({type(ex).__name__})."}


def spark_points(ds: DS.Dataset, cfg) -> tuple[list, str]:
    """The line drawn in the loading panel is the user's own data: the last observed values of the chosen target, as read
    from the file (only scaled to fit the picture). Nothing is drawn if the column cannot be used."""
    try:
        df = ds.frame
        if cfg.group_column and cfg.group_column in df.columns:
            df = df[df[cfg.group_column] == df[cfg.group_column].dropna().iloc[0]]
        vals = pd.to_numeric(df[cfg.target_column], errors="coerce").dropna()
        vals = vals[vals.abs() != float("inf")].tail(90).tolist()
        if len(vals) < 3:
            return [], ""
        return vals, f"Line: the last {len(vals)} observed values of '{cfg.target_column}' from your file, scaled to fit. It is not a forecast."
    except Exception:                                             # noqa: BLE001  (decoration only)
        return [], ""


def loader_visual_html(pts: list, note: str, rows: int, cfg, fname: str, state: str) -> str:
    W, Ht, pad = 600.0, 200.0, 18.0
    path = area = ""
    length = 0.0
    if pts:
        lo, hi = min(pts), max(pts)
        rng = (hi - lo) or 1.0
        xy = [(W * i / (len(pts) - 1), pad + (Ht - 2 * pad) * (1 - (v - lo) / rng)) for i, v in enumerate(pts)]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        area = path + f" L{W:.0f},{Ht:.0f} L0,{Ht:.0f} Z"
        length = sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(xy, xy[1:]))
    svg = (f"<svg class='ufx-spark' viewBox='0 0 {W:.0f} {Ht:.0f}' preserveAspectRatio='none' role='img' aria-label='Recent values of the target from your file' style='--len:{length + 5:.0f}'>"
           "<defs><linearGradient id='ufxg' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#ffffff' stop-opacity='.35'/><stop offset='1' stop-color='#ffffff' stop-opacity='0'/></linearGradient></defs>"
           + (f"<path class='area' d='{area}'/><path class='base' d='{path}'/><path class='line' d='{path}'/>" if path else "") + "</svg>")
    cap = f"{VM.safe(cfg.target_column)} · {VM.safe(cfg.horizon)}-step forecast · {VM.safe(fname)}"
    return (f"<div class='ufx-load {VM.safe(state)}'><div class='ufx-load-card'><div class='ufx-load-fig'>{rows:,}<small>rows in your file</small></div>"
            f"<div class='ufx-load-cap'>{cap}</div>{svg}<div class='ufx-load-note'>{VM.safe(note)}</div></div></div>")


def loader_status_html(snap: dict, target, prev_pct: int = 0, solo: bool = False) -> str:
    state, stages = snap["state"], snap["stages"]
    cur = next((i for i, x in enumerate(stages) if x["key"] == snap["stage"]), 0)
    head = {"running": f"Stage {cur + 1} of {len(stages)}", "done": "Complete", "failed": f"Stopped at stage {cur + 1} of {len(stages)}"}[state]
    label = "Forecast ready" if state == "done" else stages[cur]["label"]
    mark = {"done": "✓", "running": "•", "failed": "!", "waiting": ""}
    items = "".join(f"<li class='{x['status']}'><span class='d'>{mark[x['status']]}</span><span>{VM.safe(x['label'])}"
                    + (f"<small>took {x['seconds']:.2f} s</small>" if x["seconds"] is not None and x["status"] == "done" else
                       "<small>in progress, duration not known in advance</small>" if x["status"] == "running" else
                       "<small>stopped here</small>" if x["status"] == "failed" else "") + "</span></li>" for x in stages)
    return (f"<div role='status' aria-live='polite'><div class='ufx-band {state}{' solo' if solo else ''}'><div class='pct'>{snap['percent']}<span>%</span></div>"
            f"<div class='txt'><b>{head}</b><span>{VM.safe(label)}</span><em>{VM.safe(snap['detail'])}</em></div><span class='pill'>{VM.safe(target)}</span></div>"
            f"<div class='ufx-bar {state}' role='progressbar' aria-valuemin='0' aria-valuemax='100' aria-valuenow='{snap['percent']}'><i style='--from:{prev_pct}%;--to:{snap['percent']}%'></i></div>"
            f"<div class='ufx-basis'>{VM.safe(SV.PROGRESS_BASIS)}</div><ol class='ufx-stages'>{items}</ol></div>")


def save_run(ds, cfg, resp, explanation, seconds, error, progress):
    """Keep the run for Insights and Forecast Reports. A storage problem is reported and never hides the result."""
    try:
        rec = H.make_record(ds, cfg, resp, explanation, service.mode, seconds, error, progress)
        store().add(rec)
        S["current_run_id"] = rec["id"]
    except Exception as ex:                                       # noqa: BLE001
        SV.log.error("could not save the run record (%s) at %s", type(ex).__name__, SV.where(ex))
        S["save_warning"] = f"This run could not be saved for Insights and Forecast Reports ({type(ex).__name__}). The result below is unaffected."


def screen_processing():
    ds, cfg = S.get("dataset"), S.get("cfg")
    if ds is None or cfg is None:
        go("upload")
    if "pending_run" in S and S["pending_run"] is None:
        go("results")                                             # this submission has already been answered: never run the engine twice for one click
    run_id = S.get("pending_run")
    clear_result()
    S.pop("save_warning", None)
    key = SV.request_key(ds.sha256, cfg)                          # identity of THIS request; the result is stored under it
    S["qa"] = []
    # ---- live progress panel: every change below is triggered by a real pipeline event (SV.emit), never by a clock
    n_models = len(set(cfg.allowed_models if cfg.allowed_models is not None else SV.ROUTED_MODELS) | {"naive"})
    prog = SV.RunProgress(service.mode, n_models)
    st.subheader("Preparing your forecast")
    pts, label_note = spark_points(ds, cfg)
    st.markdown(loader_visual_html(pts, label_note, len(ds.frame), cfg, ds.name, "running"), unsafe_allow_html=True)
    status_box = st.empty()
    if service.mode == "engine":
        st.caption(f"The engine is running on {'the app server' if DS.HOSTED else 'this computer'}. The run cannot be cancelled; keep this tab open until it finishes.")
    for _ in range(70):                                           # blank what is left of the previous page while this one is working
        st.empty()
    # bring the panel into view (a fixed, constant snippet: no user or file text is ever placed in it)
    st.iframe("<script>try{const d=window.parent.document;(d.querySelector('[data-testid=\"stMain\"]')||d.scrollingElement).scrollTo({top:0,behavior:'smooth'});}catch(e){}</script>", height=1)
    painted = {"at": 0.0, "sig": None, "pct": 0}

    def paint(force: bool = False):
        sig = (prog.state, prog.stage, prog.percent, prog.detail)
        now = time.perf_counter()
        if sig == painted["sig"] or (not force and sig[:2] == (painted["sig"] or (None, None))[:2] and now - painted["at"] < 0.12):
            return                                                # many small events (one per model and series): do not flood the page
        status_box.markdown(loader_status_html(prog.snapshot(), cfg.target_column, painted["pct"]), unsafe_allow_html=True)
        painted.update(at=now, sig=sig, pct=prog.percent)

    def on_event(event, info):
        before = prog.stage
        prog.on(event, info)
        changed = prog.stage != before or event in ("done", "failed", "data_ready", "validate_done")
        paint(force=changed)
        if changed and STAGE_MIN_VISIBLE_S:
            time.sleep(STAGE_MIN_VISIBLE_S)                       # display only: the stage's real event has already happened
            prog.exclude(STAGE_MIN_VISIBLE_S)                     # and the pause is not counted in the stage's measured time
            painted["paused"] = painted.get("paused", 0.0) + STAGE_MIN_VISIBLE_S

    t0 = time.perf_counter()
    resp, stored = None, False
    with SV.progress_events(on_event):
        paint(force=True)
        SV.emit("data_ready", rows=len(ds.frame), columns=ds.frame.shape[1])
        SV.emit("validate_start")
        try:
            issues = service.validate(ds.frame, cfg)
        except Exception as ex:                                   # noqa: BLE001
            issues = [{"code": "VALIDATION_FAILED", "severity": "error", "message": f"The request could not be validated ({type(ex).__name__}).", "hint": "Try again or start over."}]
        if any(str(i.get("severity")) == "error" for i in issues):
            SV.emit("failed", code=str(issues[0].get("code")))
            S.validation_issues = issues
            S["pending_run"] = None
            go("configure")
        SV.emit("validate_done")
        try:
            if service.mode != "engine":
                SV.emit("engine_start")                           # the engine service reports its own start, finish and inner steps
            resp = service.run_forecast(ds.frame, cfg)
            if service.mode != "engine":
                SV.emit("engine_done")
            SV.emit("response_checked")
            # Stored together, before any further page update, and only if this run is still the one that is owed.
            if S.get("pending_run") == run_id:
                stored = True
                S.response, S.result_key = resp, key
                S.result_meta = {"file": ds.name, "sha256": ds.sha256, "seconds": max(time.perf_counter() - t0 - painted.get("paused", 0.0), 0.0),
                                 "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": service.mode, "run": run_id}      # work time, display pauses excluded
                S["pending_run"] = None
                if any(x.get("status") == "ok" for x in (resp.get("series") or []) if isinstance(x, dict)):
                    SV.emit("explain_start")
                    S.explanation = fetch_explanation(resp)   # only a produced forecast is explained
                    SV.emit("explain_done")
                    SV.emit("done")
                else:
                    SV.emit("failed", code="NO_FORECAST")      # the engine answered, but refused the request: its reasons are shown next
            else:
                SV.emit("failed", code="SUPERSEDED")
        except SV.ServiceUnavailable as ex:
            S.run_error = {"code": ex.code, "message": ex.message, "kind": getattr(ex, "kind", "unavailable")}
            S["pending_run"] = None
            SV.emit("failed", code=ex.code)
        except Exception as ex:                                   # noqa: BLE001  (never show a stack trace or a partial result)
            S.run_error = {"code": "FORECAST_FAILED", "message": f"The forecasting service failed ({type(ex).__name__}). No forecast was produced.", "kind": "error"}
            S["pending_run"] = None
            SV.emit("failed", code="FORECAST_FAILED")
    S.run_progress = prog.snapshot()                              # kept with the result: which stages ran and how long each really took
    if stored and not VM.check_forecast_response(resp):           # a response that fails the integrity check is never saved
        err = None if prog.state == "done" else {"code": "NO_FORECAST", "message": "The engine refused the request; its reasons are listed with the run.", "kind": "error"}
        save_run(ds, cfg, resp, S.get("explanation"), (S.get("result_meta") or {}).get("seconds"), err, S.run_progress)
    elif S.get("run_error"):
        save_run(ds, cfg, None, None, None, S.run_error, S.run_progress)
    go("results")


# ====================================================================== 4. forecast results
def brief_panel_html(s: dict, u: dict, v: dict, d: dict, last_obs) -> str:
    """Right-hand summary. Every value is a field of the engine response, or the labelled comparison of two of them."""
    fc = s.get("forecast") or []
    sel = next((c for c in s.get("candidates") or [] if c.get("model") == s.get("selected_model")), {})
    ratio = sel.get("rel_mae_vs_naive")
    rows = [("Selected model", VM.safe(s.get("selected_model")), "")]
    if fc:
        rows.append(("First forecast", f"{VM.fmt_num(fc[0]['point'])}<br><small>{VM.safe(str(fc[0]['timestamp'])[:16].replace('T', ' '))}</small>", ""))
        rows.append(("Last forecast", f"{VM.fmt_num(fc[-1]['point'])}<br><small>{VM.safe(str(fc[-1]['timestamp'])[:16].replace('T', ' '))}</small>", ""))
    if d["label"] != "unavailable":
        word = {"up": "higher", "down": "lower", "flat": "unchanged"}[d["label"]]
        amt = "" if d["label"] == "flat" else f" by {VM.fmt_num(abs(d['diff']))}" + (f" ({d['pct']:+.2f}%)" if d.get("pct") is not None else "")
        rows.append(("Last forecast against the last observed value", f"{word}{amt}<br><small>last observed: {VM.fmt_num(last_obs)}</small>", {"up": "ufx-up", "down": "ufx-down"}.get(d["label"], "")))
    if sel.get("val_mae") is not None:
        rows.append(("Validation MAE", VM.fmt_num(sel["val_mae"]), ""))
    if isinstance(ratio, (int, float)):
        rows.append(("Compared with the naive baseline", f"{VM.fmt_num(ratio)} × its validation MAE", ""))
    rows.append(("Prediction intervals", "available" if u.get("status") == "available" else "unavailable", ""))
    rows.append(("Warnings for this series", str(len(s.get("issues") or [])), ""))
    body = "".join(f"<div class='ufx-kv'><span>{VM.safe(k)}</span><b class='{c}'>{val}</b></div>" for k, val, c in rows)
    note = "Validation figures describe one past period of your data. They are not test results and not a guarantee." if v else "No validation was possible on this history, so no accuracy figure exists."
    return f"<div class='ufx-panel'><h4>Forecast brief</h4>{body}<div class='ufx-basis' style='margin:.7rem 0 0 0'>{VM.safe(note)}</div></div>"


def model_bars_html(s: dict) -> str:
    ev = [c for c in s.get("candidates") or [] if c.get("val_mae") is not None]
    if not ev:
        return "<div class='ufx-panel'><h4>Model comparison</h4><div class='ufx-basis'>No model was scored on a validation period for this series.</div></div>"
    top = max(c["val_mae"] for c in ev) or 1.0
    rows = "".join(f"<div class='ufx-row {'sel' if c.get('model') == s.get('selected_model') else ''}'><span class='nm'>{VM.safe(c.get('model'))}</span>"
                   f"<span class='tr'><i style='width:{max(100.0 * c['val_mae'] / top, 2):.1f}%'></i></span><span class='val'>{VM.fmt_num(c['val_mae'])}</span></div>" for c in ev)
    return f"<div class='ufx-panel'><h4>Model comparison · validation MAE</h4>{rows}<div class='ufx-basis' style='margin:.6rem 0 0 0'>Shorter bar = lower error on the validation period. The selected model is white.</div></div>"


def history_tiles_html(s: dict) -> str:
    h = s.get("history") or {}
    cells = [("Observed points", VM.fmt_num(h.get("n_observed"))), ("Missing points", VM.fmt_num(h.get("n_missing"))),
             ("History starts", VM.safe(str(h.get("start", "unavailable"))[:16].replace("T", " "))), ("History ends", VM.safe(str(h.get("end", "unavailable"))[:16].replace("T", " ")))]
    body = "".join(f"<div class='ufx-mini'><div class='k'>{k}</div><div class='v'>{val}</div></div>" for k, val in cells)
    return f"<div class='ufx-panel'><h4>History used by the engine</h4><div class='ufx-grid2'>{body}</div></div>"


def notices_html(resp: dict, s: dict) -> str:
    items = [(str(i.get("severity")), f"{i.get('code')}: {i.get('message')}") for i in (resp.get("issues") or []) + (s.get("issues") or [])]
    body = "".join(f"<div class='ufx-li'><span class='ufx-tag {VM.safe(sev)}'>{VM.safe(sev)}</span><span>{VM.safe(msg, 220)}</span></div>" for sev, msg in items[:5])
    if not items:
        body = "<div class='ufx-li'><span class='ufx-tag'>clear</span><span>The engine reported no warnings for this series.</span></div>"
    n_lim = len(s.get("limitations") or [])
    body += f"<div class='ufx-li'><span class='ufx-tag'>limits</span><span>{n_lim} limitation{'s' if n_lim != 1 else ''} reported by the engine. See the Warnings and limitations tab.</span></div>"
    return f"<div class='ufx-panel'><h4>Warnings and limits</h4>{body}</div>"


def nav_buttons(prefix: str, saved: bool = False):
    b = st.columns([1.5, 1.2, 1.4, 3])
    if not saved and b[0].button("Adjust configuration", key=f"{prefix}_back", width="stretch"):
        go("configure")
    if b[1].button("New forecast", key=f"{prefix}_new", width="stretch"):
        reset_all()
    if S.get("current_run_id") or saved:
        if b[2].button("Open its report", key=f"{prefix}_report", width="stretch"):
            S["report_pick"] = (S.get("view_record") or {}).get("id") or S.get("current_run_id")
            go("reports")


def screen_results():
    rec = S.get("view_record")                                    # a saved run opened from Insights or Forecast Reports
    if rec is not None:
        ds, resp = None, rec["response"]
        cfg = SV.ForecastConfig(rec["config"]["timestamp_column"], rec["config"]["target_column"], rec["config"]["horizon"], rec["config"]["group_column"],
                                rec["config"]["frequency"], rec["config"]["allowed_models"], confirmed=True)
    else:
        ds, cfg, resp = S.get("dataset"), S.get("cfg"), S.get("response")
        if resp is not None and "result_key" in S and (ds is None or cfg is None or S.result_key != SV.request_key(ds.sha256, cfg)):
            clear_result()                                        # never show a result under a different file or different settings
            resp = None
            st.info("The dataset or settings changed after the last run, so the earlier result was discarded. Run the forecast again.")
    if S.get("save_warning"):
        st.warning(S["save_warning"])
    if resp is None:
        e = S.get("run_error")
        if e:
            (st.info if e["kind"] == "unavailable" else st.error)(f"{e['code']}: {e['message']}")
        snap = S.get("run_progress")
        if e and snap and snap.get("state") == "failed" and cfg is not None:
            st.markdown(loader_status_html(snap, cfg.target_column, snap.get("percent", 0), solo=True), unsafe_allow_html=True)      # where the run stopped; nothing is animated
        UI.empty_state("No forecast to show", "No forecast was computed, so there are no forecast values, validation metrics, intervals or explanation. Nothing is displayed in their place.")
        if e and ds is not None and cfg is not None and st.button("Try again", key="empty_retry", type="primary", help="Runs the same request again."):
            start_run()
        nav_buttons("empty")
        return
    problems = VM.check_forecast_response(resp)
    if problems:
        st.error("The forecasting service returned a response that failed the interface's integrity checks, so it is not displayed.")
        for p in problems:
            st.write("• " + VM.safe_md(p))
        nav_buttons("bad")
        return
    series = resp.get("series") or []
    if resp.get("status") == "error" or not any(x.get("status") == "ok" for x in series):
        st.subheader("No forecast was produced")
        st.error("The forecasting engine could not forecast this request. Its reasons are listed below; no values are shown or estimated.")
        UI.issues_list(resp.get("issues") or [])
        seen = {(i.get("code"), i.get("message")) for i in resp.get("issues") or []}
        UI.issues_list([i for x in series for i in (x.get("issues") or []) if (i.get("code"), i.get("message")) not in seen])
        UI.empty_state("No forecast to show", "Go back to the configuration, fix what the engine reported and run again.")
        nav_buttons("noseries")
        return
    st.subheader("Forecast results")
    if rec is not None:
        st.caption(f"Saved run from {rec['created']}: file '{VM.safe_md(rec['dataset']['name'])}', target '{VM.safe_md(cfg.target_column)}'."
                   + (" DEMO: stored result for the synthetic sample." if rec.get("mode") == "demo" else "")
                   + " The chart shows the stored recent values; the uploaded file itself was not kept.")
        if S.get("response") is not None and st.button("Return to the current result", key="btn_leave_saved"):
            S.pop("view_record", None)
            st.rerun()
    else:
        meta = S.get("result_meta") or {}
        if meta:
            st.caption(f"Result for file '{VM.safe_md(meta.get('file'))}' (fingerprint {str(meta.get('sha256'))[:12]}), target '{VM.safe_md(cfg.target_column if cfg else '')}', "
                       f"finished {meta.get('finished')} after {meta.get('seconds', 0):.1f} s." + (" DEMO: stored result for the synthetic sample." if meta.get("mode") == "demo" else ""))
    UI.issues_list(resp.get("issues") or [])
    labels = [f"{s.get('target')}" + (f" · {s.get('group')}" if s.get("group") is not None else "") for s in series]
    idx = st.selectbox("Series", range(len(series)), format_func=lambda i: labels[i], key="sel_series") if len(series) > 1 else 0
    s = series[idx]
    if s.get("status") != "ok" or not s.get("forecast"):
        st.error("The engine did not produce a forecast for this series.")
        UI.issues_list(s.get("issues") or [])
        nav_buttons("serr", saved=rec is not None)
        return
    k = st.columns(5)
    UI.kpi(k[0], "Selected model", s.get("selected_model"), "chosen by the engine")
    UI.kpi(k[1], "Horizon", f"{len(s['forecast'])} steps", f"frequency {s.get('frequency')} ({s.get('frequency_source')})")
    UI.kpi(k[2], "Forecast origin", str(s.get("forecast_origin", ""))[:16].replace("T", " "), "last observation used")
    v = s.get("validation") or {}
    UI.kpi(k[3], "Validation origins", VM.fmt_num(v.get("n_common_origins")) if v else "none", "same origins for every model" if v else "no validation was possible")
    u = s.get("uncertainty") or {}
    UI.kpi(k[4], "Prediction intervals", "available" if u.get("status") == "available" else "unavailable", "from validation residuals" if u.get("status") == "available" else "see below")

    # ---- chart: observed history + the engine's forecast points, unchanged
    fc = VM.forecast_frame(s)
    levels = VM.interval_levels(s)
    if rec is not None:
        hist = H.tail_frame(rec, s.get("group"))
    elif ds is not None and cfg and s.get("target") in ds.frame.columns:
        hist = VM.history_frame(ds.frame, cfg.timestamp_column, s.get("target"), cfg.group_column, s.get("group"))
    else:
        hist = pd.DataFrame(columns=["timestamp", "value", "kind"])
    seen_vals = hist["value"].dropna() if len(hist) else pd.Series(dtype=float)
    last_obs = float(seen_vals.iloc[-1]) if len(seen_vals) else None
    d = R.direction(last_obs, s["forecast"][-1]["point"])         # the one colour rule: forecast end against the last observed value
    main, side = st.columns([2.35, 1], gap="medium")
    with main:
        last_row = hist.tail(1)
        hist, kthin = VM.thin(hist)
        if kthin > 1 and len(last_row) and hist.index[-1] != last_row.index[-1]:
            hist = pd.concat([hist, last_row])                    # display thinning must keep the last observed value the forecast starts from
        with st.container(key="ufa_chart"):
            st.iframe(CH.forecast_chart_src(hist, s, s.get("target"), d["label"], max(max(10 * len(fc), 120) // max(kthin, 1), 10)), height=486)
        col = {"up": "green: its last value is above the last observed value", "down": "red: its last value is below the last observed value",
               "flat": "white: its last value equals the last observed value"}.get(d["label"], "white: no observed value is available to compare it with")
        legend = f"Gray line: observed values. Line with points: the engine's forecast, drawn in {col}."
        legend += (" Gray bands: prediction intervals (" + ", ".join(f"{float(l):.0%}" for l in levels) + ") returned by the engine.") if levels else " No interval band is drawn because the engine returned none."
        if kthin > 1:
            legend += f" For display, every {kthin}th observed point is drawn; the data are unchanged."
        st.caption(legend)
        if not levels:
            st.info("Prediction intervals are unavailable: " + str(u.get("reason") or "the engine did not report a reason") + ". No bounds are shown or estimated.")
    with side:
        st.markdown(brief_panel_html(s, u, v, d, last_obs), unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3, gap="medium")
    b1.markdown(model_bars_html(s), unsafe_allow_html=True)
    b2.markdown(history_tiles_html(s), unsafe_allow_html=True)
    b3.markdown(notices_html(resp, s), unsafe_allow_html=True)
    snap = (rec or {}).get("progress") if rec is not None else S.get("run_progress")
    if snap and snap.get("state") == "done":
        with st.expander("How this result was produced (stages that actually ran, with measured times)"):
            st.markdown(loader_status_html(snap, s.get("target"), 100, solo=True), unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Forecast values", "Model selection and validation", "Warnings and limitations", "Explanation"])
    with t1:
        tbl = fc.drop(columns=["kind"]).rename(columns={"timestamp": "Time", "value": "Forecast", "step": "Step"})
        names = {VM.level_key(lv): f"{float(lv):.0%}" for lv in levels}
        tbl = tbl.rename(columns={c: (f"Lower {names.get(c[3:], c[3:])}" if c.startswith("lo_") else f"Upper {names.get(c[3:], c[3:])}") for c in tbl.columns if c.startswith(("lo_", "hi_"))})
        st.dataframe(tbl[["Step", "Time", "Forecast"] + [c for c in tbl.columns if c.startswith(("Lower", "Upper"))]], width="stretch", hide_index=True)
        st.caption("Values exactly as returned by the forecasting engine.")
    with t2:
        st.markdown(UI.badge("Engine output", "engine"), unsafe_allow_html=True)
        st.markdown("**Why this model:** " + VM.safe_md(s.get("selection_reason")))
        ct = VM.candidates_table(s)
        st.dataframe(ct, width="stretch", hide_index=True)
        st.caption("Validation metrics: measured on a validation period inside your data (same forecast origins for every model, in the target's own units; lower is better). "
                   "They are not test results, and past validation accuracy does not guarantee future accuracy. "
                   "An empty or 'None' cell means the model was not evaluated on this series; the Status and Note columns say why.")
        tb = s.get("training_boundary") or {}
        if v or tb:
            b = st.columns(3)
            UI.kpi(b[0], "Model selection used data up to", str(tb.get("selection_train_end", "unavailable"))[:16].replace("T", " "), "training boundary")
            UI.kpi(b[1], "Validation period", (str(v.get("first_origin", ""))[:16].replace("T", " ") + " → " + str(v.get("last_origin", ""))[:16].replace("T", " ")) if v else "unavailable", "forecast origins")
            UI.kpi(b[2], "Final fit used data up to", str(tb.get("final_fit_end", "unavailable"))[:16].replace("T", " "), "after selection")
    with t3:
        iss = s.get("issues") or []
        if iss:
            UI.issues_list(iss)
        else:
            st.caption("The engine reported no warnings for this series.")
        st.markdown("**Limitations reported by the engine**")
        for lim in s.get("limitations") or ["none reported"]:
            st.write("• " + VM.safe_md(lim))
        if u.get("status") == "available":
            st.markdown("**About the intervals**")
            for c in u.get("caveats") or []:
                st.write("• " + VM.safe_md(c))
    with t4:
        explanation_panel(resp, s, (rec or {}).get("explanation") if rec is not None else None)
    nav_buttons("res", saved=rec is not None)


def explanation_panel(resp: dict, s: dict, saved: dict | None = None):
    st.markdown(UI.badge("Numbers: forecasting engine", "engine") + UI.badge("Words: explanation layer", "words"), unsafe_allow_html=True)
    if saved is not None:
        ex = saved
    else:
        if "explanation" not in S:
            S.explanation = fetch_explanation(resp)
        ex = S.explanation
    if ex.get("_unavailable"):
        st.info(ex["_unavailable"])
        UI.empty_state("No explanation available", "The forecast above is unaffected. No explanatory text has been generated in its place.")
        return
    probs = VM.check_explanation(ex, s)
    if probs or ex.get("status") != "ok":
        st.error("The explanation was not shown because it failed the interface's checks: " + "; ".join(probs or [str(ex.get("error_code"))]))
        return
    src = {"deterministic_template": "Deterministic template: generated by fixed rules from the engine output. No language model was used.",
           "deterministic_fallback": "Deterministic template (fallback): language-model wording was unavailable or rejected by the grounding checks.",
           "llm_validated": "Language-model wording that passed the grounding checks. Numbers come from the engine."}[ex["source"]]
    st.caption(src)
    for title, key in (("Summary", "summary"), ("Trend", "trend_interpretation"), ("Model selection", "model_selection_explanation"), ("Uncertainty", "uncertainty_explanation")):
        st.markdown(f"**{title}**")
        st.write(VM.safe_md(ex.get(key)))
    if ex.get("unavailable_evidence"):
        st.markdown("**Not available**")
        for x in ex["unavailable_evidence"]:
            st.write("• " + VM.safe_md(x))
    with st.expander("Evidence references (fields of the engine response behind each section)"):
        for r in ex.get("evidence_references") or []:
            st.write(VM.safe_md(f"{r.get('section')}: " + ", ".join(map(str, r.get("paths") or []))))
    st.markdown("<div class='ufa-section'></div>", unsafe_allow_html=True)
    st.markdown("**Ask about this forecast**")
    st.caption("Answers use only this forecast response. Questions that need new numbers (another horizon, target or model) are answered with how to run a new forecast.")
    def _use_suggestion():                                        # a keyed text input ignores `value` after its first render
        if S.get("pill_q"):
            S["txt_q"] = S["pill_q"]
    if hasattr(st, "pills"):
        st.pills("Suggested questions", SUGGESTED, key="pill_q", on_change=_use_suggestion)
    q = st.text_input("Your question", max_chars=1000, key="txt_q", placeholder="e.g. Why was this model selected?")
    if st.button("Ask", key="btn_ask", disabled=not (q or "").strip()):
        try:
            a = service.explain(resp, q.strip(), "concise")
            S.setdefault("qa", []).append({"q": q.strip(), "a": a})
        except SV.ServiceUnavailable as e2:
            S.setdefault("qa", []).append({"q": q.strip(), "err": f"{e2.code}: {e2.message}"})
        except Exception as e2:                                   # noqa: BLE001
            S.setdefault("qa", []).append({"q": q.strip(), "err": f"The question could not be answered ({type(e2).__name__})."})
    for item in reversed(S.get("qa") or []):
        st.markdown("**Q:** " + VM.safe_md(item["q"]))
        if item.get("err"):
            st.warning(item["err"])
            continue
        a = item["a"]
        if VM.check_explanation(a, s) or a.get("status") != "ok":
            st.warning("The answer was not shown because it failed the interface's checks.")
            continue
        qd = a.get("question") or {}
        if qd.get("requires_new_forecast"):
            st.info("This needs a new forecast. " + str(qd.get("new_request_hint") or ""))
        elif not qd.get("answerable"):
            st.warning("Not answerable from this forecast" + (": missing " + ", ".join(map(str, qd.get("missing_evidence") or [])) if qd.get("missing_evidence") else "."))
        st.write(VM.safe_md(qd.get("answer")))


# ====================================================================== 5. insights
def run_label(r: dict) -> str:
    return f"{r['created']} · {r['dataset']['name']} · {r['config']['target_column']}" + (" · DEMO" if r.get("mode") == "demo" else "") + (" · failed" if r["status"] != "ok" else "")


def dir_cell(d: dict, baseline: str) -> str:
    if d["label"] == "unavailable":
        return "<small>unavailable</small>"
    cls = {"up": "ufx-up", "down": "ufx-down"}.get(d["label"], "")
    word = {"up": "▲ higher", "down": "▼ lower", "flat": "unchanged"}[d["label"]]
    amt = "" if d["label"] == "flat" else f" by {VM.fmt_num(abs(d['diff']))}" + (f" ({d['pct']:+.1f}%)" if d.get("pct") is not None else "")
    return f"<span class='{cls}'>{word}{amt}</span><br><small>{VM.safe(baseline)}</small>"


def series_rows(records: list[dict]) -> list[dict]:
    """One row per forecast series (or per failed run), every value read from the saved record."""
    out = []
    for r in records:
        c, dsm = r["config"], r["dataset"]
        base = {"run_id": r["id"], "Run": r["created"], "Dataset": dsm["name"], "Target": c["target_column"], "Mode": "DEMO" if r.get("mode") == "demo" else "live",
                "epoch": r.get("created_epoch", 0)}
        if r["status"] != "ok":
            out.append(base | {"Series": "", "Model": "", "Horizon": c.get("horizon"), "Validation MAE": None, "MAE ÷ naive": None, "Forecast min": None, "Forecast max": None,
                               "Last observed": None, "Last forecast": None, "Change vs last observed (%)": None, "Direction": "no forecast", "Intervals": "",
                               "Warnings": None, "Status": "failed: " + str((r.get("error") or {}).get("code") or "no forecast")})
            continue
        for s in R.ok_series(r):
            f = R.series_facts(r, s)
            d = f["direction"]
            out.append(base | {"Series": "" if f["group"] is None else str(f["group"]), "Model": f["selected_model"], "Horizon": f["horizon"], "Validation MAE": f["val_mae"],
                               "MAE ÷ naive": f["rel_mae_vs_naive"], "Forecast min": f["forecast_min"], "Forecast max": f["forecast_max"], "Last observed": f["last_observed"],
                               "Last forecast": f["last_forecast"]["point"], "Change vs last observed (%)": d.get("pct"), "Direction": d["label"],
                               "Intervals": "yes" if f["intervals"] else "no", "Warnings": len(f["issues"]) + len(r["response"].get("issues") or []), "Status": "completed"})
    return out


def go_to_run(rec: dict):
    for kx in ("explanation", "qa") + WIDGET_RESULT_KEYS:
        S.pop(kx, None)
    S["view_record"] = rec
    go("results")


def table_pick(event, rows: list[dict], state_key: str, memo_key: str):
    """A row chosen in a table selects the same run in the picker below (only when the table selection changes)."""
    try:
        sel = tuple(event.selection.rows)
    except Exception:                                             # noqa: BLE001
        sel = ()
    if sel and S.get(memo_key) != sel and sel[0] < len(rows):
        S[state_key] = rows[sel[0]]["run_id"]
    S[memo_key] = sel


def screen_insights():
    kept = store()
    records = kept.all()
    st.markdown("<div class='ufx-h'><h2>Insights</h2><p>A workspace over every saved forecast. Every figure is read from a saved run; nothing is estimated, "
                "and runs are compared only when they forecast the same data.</p></div>", unsafe_allow_html=True)
    where = f"saved on this computer in '{kept.location}'" if kept.kind == "folder" else "from this browser session only (nothing is saved to disk in this mode)"
    st.caption(f"Built only from completed forecast runs {where}. Nothing here is estimated or assumed." + (" " + HOSTED_HISTORY_NOTE if DS.HOSTED else ""))
    if kept.skipped():
        st.warning(f"{kept.skipped()} saved file(s) could not be read or failed the integrity check and are not included.")
    if not records:
        UI.empty_state("No forecasts yet", "Run a forecast and it will be summarised here: models chosen, validation results, forecast direction and warnings.")
        _, b, _ = st.columns([1, 1.2, 1])
        if b.button("Upload a dataset to start", key="ins_start", type="primary", icon=":material/upload_file:", width="stretch"):
            go("upload")
        return
    # ---- filters (all optional; an empty filter means "everything")
    f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.1, 1.3])
    names = sorted({r["dataset"]["name"] for r in records})
    targets = sorted({str(r["config"]["target_column"]) for r in records})
    models = sorted({str(s.get("selected_model")) for r in records for s in R.ok_series(r)})
    days = sorted({pd.Timestamp(r["created"]).date() for r in records})
    fd = f1.multiselect("Datasets", names, key="ins_f_ds", placeholder="All datasets")
    ft = f2.multiselect("Targets", targets, key="ins_f_tgt", placeholder="All targets")
    fm = f3.multiselect("Models", models, key="ins_f_model", placeholder="All models")
    rng = f4.date_input("Runs between", value=(days[0], days[-1]), min_value=days[0], max_value=days[-1], key="ins_f_dates")
    lo_d, hi_d = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (days[0], days[-1]))
    chosen = [r for r in records if (not fd or r["dataset"]["name"] in fd) and (not ft or str(r["config"]["target_column"]) in ft)
              and lo_d <= pd.Timestamp(r["created"]).date() <= hi_d
              and (not fm or any(str(s.get("selected_model")) in fm for s in R.ok_series(r)))]
    if len(chosen) != len(records):
        st.caption(f"Showing {len(chosen)} of {len(records)} saved runs.")
    if not chosen:
        UI.empty_state("No runs match these filters", "Clear a filter to see more runs.")
        return
    o = R.overview(chosen)
    k = st.columns(5)
    UI.kpi(k[0], "Forecast runs", f"{o['runs']:,}", f"{o['completed']} completed, {o['failed']} without a forecast")
    UI.kpi(k[1], "Datasets", f"{o['datasets']:,}", "distinct files, by content")
    UI.kpi(k[2], "Series forecast", f"{o['series']:,}", "across completed runs")
    UI.kpi(k[3], "Forecast ends higher / lower", f"{o['directions']['up']} / {o['directions']['down']}", "against each series' last observed value")
    UI.kpi(k[4], "Without intervals", f"{o['without_intervals']:,}", "series where the engine returned none")
    if o["demo_runs"]:
        st.info(f"{o['demo_runs']} of these runs are DEMO output (a stored result for the synthetic sample), not forecasts of your data.")
    rows = series_rows(chosen)
    if fm:
        rows = [x for x in rows if x["Model"] in fm or x["Status"] != "completed"]
    done = [x for x in rows if x["Status"] == "completed"]
    # ---- notable facts: each one is a plain reading of saved numbers, shown only when the numbers exist
    facts = []
    moves = [x for x in done if isinstance(x["Change vs last observed (%)"], (int, float))]
    if moves:
        m = max(moves, key=lambda x: abs(x["Change vs last observed (%)"]))
        cls = {"up": "ufx-up", "down": "ufx-down"}.get(m["Direction"], "")
        facts.append(("Largest projected change", f"<span class='big {cls}'>{m['Change vs last observed (%)']:+.2f}%</span><br>{VM.safe(m['Target'])}"
                      + (f" · {VM.safe(m['Series'])}" if m["Series"] else "") + f" in {VM.safe(m['Dataset'])}<br><small>last forecast against the last observed value, run {VM.safe(m['Run'])}</small>"))
    behind = [x for x in done if isinstance(x["MAE ÷ naive"], (int, float)) and x["MAE ÷ naive"] >= 1]
    if done:
        facts.append(("Against the naive baseline", f"<span class='big'>{len(done) - len(behind)} of {len(done)}</span><br>series where the selected model had a lower validation error "
                      "than repeating the last value" + (f"; in {len(behind)} it did not, so read those forecasts with extra care" if behind else "")))
    noint = [x for x in done if x["Intervals"] == "no"]
    if noint:
        facts.append(("No prediction intervals", f"<span class='big'>{len(noint)}</span><br>series where the engine returned no interval, usually because the history was too short"))
    failed = [x for x in rows if x["Status"] != "completed"]
    if failed:
        facts.append(("Runs without a forecast", f"<span class='big'>{len(failed)}</span><br>most recent: {VM.safe(failed[0]['Status'][8:])} ({VM.safe(failed[0]['Run'])})"))
    if facts:
        st.markdown("<div class='ufx-facts'>" + "".join(f"<div class='ufx-fact' style='--i:{i}'><b>{VM.safe(t)}</b>{h}</div>" for i, (t, h) in enumerate(facts)) + "</div>",
                    unsafe_allow_html=True)
    # ---- interactive charts of saved numbers
    if done:
        cdf = pd.DataFrame(done)
        cdf["Label"] = cdf["Run"].str.slice(5, 16) + " · " + cdf["Target"].astype(str) + cdf["Series"].map(lambda g: f" · {g}" if g else "")
        with st.container(key="ufa_ins_charts"):
            ch1, ch2 = st.columns(2, gap="medium")
            hover = alt.selection_point(on="pointerover", fields=["Label"], empty=True, clear="pointerout")
            tip = [alt.Tooltip("Run:N"), alt.Tooltip("Dataset:N"), alt.Tooltip("Target:N"), alt.Tooltip("Series:N"), alt.Tooltip("Model:N"),
                   alt.Tooltip("Last observed:Q", format=",.6~g"), alt.Tooltip("Last forecast:Q", format=",.6~g"), alt.Tooltip("Change vs last observed (%):Q", format="+.2f")]
            mv = cdf.dropna(subset=["Change vs last observed (%)"])
            if len(mv):
                bars = alt.Chart(mv).mark_bar(cornerRadiusEnd=4, size=18).encode(
                    y=alt.Y("Label:N", title=None, sort=None, axis=alt.Axis(labelColor=UI.C["muted"], labelLimit=220)),
                    x=alt.X("Change vs last observed (%):Q", title="Last forecast against the last observed value (%)", axis=alt.Axis(labelColor=UI.C["muted"], titleColor=UI.C["muted"], gridColor="#1c1c1e")),
                    color=alt.Color("Direction:N", scale=alt.Scale(domain=["up", "down", "flat"], range=[UI.C["pos"], UI.C["neg"], "#d4d4d8"]), legend=None),
                    opacity=alt.condition(hover, alt.value(1), alt.value(.35)), tooltip=tip).add_params(hover)
                zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6b6b70").encode(x="x:Q")
                ch1.markdown("**Projected change by series**")
                ch1.altair_chart((zero + bars).properties(height=max(170, 46 * len(mv)), background="transparent").configure_view(stroke=None), width="stretch")
                ch1.caption("Green: the forecast ends above the last observed value; red: below. The baseline is each series' own last observed value.")
            ev = cdf.dropna(subset=["MAE ÷ naive"])
            if len(ev):
                pts = alt.Chart(ev).mark_circle(size=120, color="#ffffff", stroke="#000", strokeWidth=1).encode(
                    x=alt.X("Label:N", title=None, sort=None, axis=alt.Axis(labelAngle=-30, labelColor=UI.C["muted"], labelLimit=160)),
                    y=alt.Y("MAE ÷ naive:Q", title="Validation MAE ÷ naive MAE", scale=alt.Scale(zero=True), axis=alt.Axis(labelColor=UI.C["muted"], titleColor=UI.C["muted"], gridColor="#1c1c1e")),
                    opacity=alt.condition(hover, alt.value(1), alt.value(.35)), size=alt.condition(hover, alt.value(220), alt.value(120)),
                    tooltip=tip[:5] + [alt.Tooltip("Validation MAE:Q", format=",.6~g"), alt.Tooltip("MAE ÷ naive:Q", format=".4f")]).add_params(hover)
                one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#6b6b70", strokeDash=[4, 4]).encode(y="y:Q")
                ch2.markdown("**Validation error against the naive baseline**")
                ch2.altair_chart((one + pts).properties(height=280, background="transparent").configure_view(stroke=None), width="stretch")
                ch2.caption("Below the dashed line (1.0) the selected model beat 'repeat the last value' on the validation period. Values are in no single unit, so different targets can share this axis.")
    # ---- every run, as a sortable table; picking a row selects the run below
    st.markdown("**Every run**")
    show = pd.DataFrame(rows).drop(columns=["run_id", "epoch"])
    with st.container(key="ufa_ins_table"):
        ev_t = st.dataframe(show, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key="ins_table",
                            column_config={"Validation MAE": st.column_config.NumberColumn(format="%.6g"), "MAE ÷ naive": st.column_config.NumberColumn(format="%.4f"),
                                           "Forecast min": st.column_config.NumberColumn(format="%.6g"), "Forecast max": st.column_config.NumberColumn(format="%.6g"),
                                           "Last observed": st.column_config.NumberColumn(format="%.6g"), "Last forecast": st.column_config.NumberColumn(format="%.6g"),
                                           "Change vs last observed (%)": st.column_config.NumberColumn(format="%+.2f")})
    table_pick(ev_t, rows, "ins_pick", "_ins_table_sel")
    st.markdown("<div class='ufx-basis'>Green and red mark direction only: a value above or below the baseline named with it. "
                "They say nothing about whether a change is good or bad. Validation MAE is in each target's own units, so it is not comparable between different targets. "
                + VM.safe(R.OTHER_COLUMNS_NOTE) + "</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    mrows = "".join(f"<div class='ufx-kv'><span>{VM.safe(m)}</span><b>{n} series</b></div>" for m, n in sorted(o["models"].items(), key=lambda x: -x[1]))
    note = (f"In {o['naive_selected']} series no model beat the simplest method (repeat the last value) by a clear margin, so the engine kept it." if o["naive_selected"] else
            "In every series at least one model beat the simplest method on the validation period.")
    c1.markdown(f"<div class='ufx-panel'><h4>Models the engine selected</h4>{mrows or '<div class=ufx-basis>No completed run.</div>'}<div class='ufx-basis' style='margin:.7rem 0 0 0'>{VM.safe(note)}</div></div>",
                unsafe_allow_html=True)
    irows = "".join(f"<div class='ufx-kv'><span>{VM.safe(code)}</span><b>{n}</b></div>" for code, n in sorted(o["issue_codes"].items(), key=lambda x: -x[1]))
    c2.markdown("<div class='ufx-panel'><h4>Warnings the engine raised</h4>" + (irows or "<div class='ufx-basis'>None in any completed run.</div>")
                + "<div class='ufx-basis' style='margin:.7rem 0 0 0'>Counts of warning codes across completed runs. Open a run to read each message.</div></div>", unsafe_allow_html=True)
    groups = R.comparable_groups(chosen)
    st.markdown("<div class='ufa-section'></div>", unsafe_allow_html=True)
    st.markdown("**Comparisons between runs**")
    if not groups:
        st.caption("No two completed runs forecast the same file and target, so no run-to-run comparison is shown. Runs on different data or targets are not combined.")
    for g in groups:
        first = g[0]
        body = ""
        for r in g:
            for s in R.ok_series(r):
                f = R.series_facts(r, s)
                body += (f"<tr><td>{VM.safe(r['created'])}</td><td>{VM.safe(f['group']) if f['group'] is not None else 'single series'}</td><td>{f['horizon']} steps</td>"
                         f"<td>{VM.safe(f['frequency'])}</td><td>{VM.safe(f['selected_model'])}</td><td>{VM.fmt_num(f['val_mae'])}</td><td>{VM.fmt_num(f['rel_mae_vs_naive'])}</td>"
                         f"<td>{VM.fmt_num(f['last_forecast']['point'])}</td></tr>")
        st.markdown(f"<div class='ufx-panel'><h4>Same file and target: {VM.safe(first['dataset']['name'])} · {VM.safe(first['config']['target_column'])} · {len(g)} runs</h4>"
                    "<div style='overflow-x:auto'><table class='ufx-table'><tr><th>Run</th><th>Series</th><th>Horizon</th><th>Frequency</th><th>Model</th><th>Validation MAE</th>"
                    f"<th>MAE ÷ naive</th><th>Last forecast</th></tr>{body}</table></div><div class='ufx-basis' style='margin:.7rem 0 0 0'>Same data and target, so the rows can be read "
                    "side by side. A validation MAE at a longer horizon covers more steps and is not strictly like-for-like with a shorter one.</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='ufa-section'></div>", unsafe_allow_html=True)
    ids = list(dict.fromkeys(x["run_id"] for x in rows))
    if S.get("ins_pick") not in ids:
        S.pop("ins_pick", None)
    pick = st.selectbox("Open a run", ids, format_func=lambda i: run_label(next(r for r in chosen if r["id"] == i)), key="ins_pick",
                        help="Or choose a row in the table above.")
    b = st.columns([1.6, 1.6, 4])
    rec = next(r for r in chosen if r["id"] == pick)
    if b[0].button("Open forecast result", key="ins_open", icon=":material/show_chart:", disabled=rec["status"] != "ok", width="stretch"):
        go_to_run(rec)
    if b[1].button("Open its report", key="ins_report", icon=":material/description:", width="stretch"):
        S["report_pick"] = pick
        go("reports")


# ====================================================================== 6. forecast reports
def _toast(name: str):
    def cb():
        st.toast(f"Download prepared: {name}", icon=":material/download_done:")
    return cb


def screen_reports():
    kept = store()
    records = kept.all()
    st.markdown("<div class='ufx-h'><h2>Forecast reports</h2><p>Every saved forecast, with its files. Preview a report before you download it.</p></div>", unsafe_allow_html=True)
    where = (f"Reports are kept on this computer in '{kept.location}', one file per run. Nothing is overwritten; delete a report here to remove it."
             if kept.kind == "folder" else HOSTED_HISTORY_NOTE if DS.HOSTED else "Reports are kept for this browser session only; they are gone when the session ends.")
    st.caption(where)
    if S.get("_rep_flash"):
        st.toast(S.pop("_rep_flash"), icon=":material/check:")
    if kept.skipped():
        st.warning(f"{kept.skipped()} saved file(s) could not be read or failed the integrity check and are not listed.")
    if not records:
        UI.empty_state("No reports yet", "Each completed forecast gets a report here, with CSV and HTML downloads.")
        _, b, _ = st.columns([1, 1.2, 1])
        if b.button("Upload a dataset to start", key="rep_start", type="primary", icon=":material/upload_file:", width="stretch"):
            go("upload")
        return
    ids_all = [r["id"] for r in records]
    if S.get("report_pick") in ids_all:                            # arrived from a result or from Insights: show that report, whatever the filters were
        S["rep_pick"] = S.pop("report_pick")
        S["rep_search"], S["rep_show"] = "", "All"
    c1, c2 = st.columns([2.2, 1.6])
    q = c1.text_input("Search reports", key="rep_search", placeholder="Dataset, target, model or run id", icon=":material/search:")
    show = c2.segmented_control("Show", ["All", "Completed", "Failed", "Demo"], key="rep_show", default="All") or "All"
    ql = (q or "").strip().lower()

    def keep(r):
        hay = " ".join([r["id"], r["dataset"]["name"], str(r["config"]["target_column"])] + [str(s.get("selected_model")) for s in R.ok_series(r)]).lower()
        kind = {"All": True, "Completed": r["status"] == "ok", "Failed": r["status"] != "ok", "Demo": r.get("mode") == "demo"}[show]
        return kind and (not ql or ql in hay)
    shown = [r for r in records if keep(r)]
    if not shown:
        UI.empty_state("No reports match", "Change the search or the filter to see more reports.")
        return
    lib = pd.DataFrame([{"Generated": r["created"], "Dataset": r["dataset"]["name"], "Target": r["config"]["target_column"],
                         "Model": ", ".join(dict.fromkeys(str(s.get("selected_model")) for s in R.ok_series(r))) or "—", "Horizon": f"{r['config']['horizon']} steps",
                         "Mode": "DEMO" if r.get("mode") == "demo" else "live", "Status": "completed" if r["status"] == "ok" else "no forecast"} for r in shown])
    with st.container(key="ufa_rep_list"):
        ev_t = st.dataframe(lib, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key="rep_table", height=min(38 + 35 * len(lib), 320))
    table_pick(ev_t, [{"run_id": r["id"]} for r in shown], "rep_pick", "_rep_table_sel")
    ids = [r["id"] for r in shown]
    if S.get("rep_pick") not in ids:
        S.pop("rep_pick", None)
    pick = st.selectbox("Report", ids, format_func=lambda i: run_label(next(r for r in shown if r["id"] == i)), key="rep_pick", help="Or choose a row in the library above.")
    r = next(x for x in shown if x["id"] == pick)
    c, dsm = r["config"], r["dataset"]
    if r.get("mode") == "demo":
        st.warning("DEMO OUTPUT: this report is a stored result for the bundled synthetic sample, not a forecast of your data.")
    chips = [f"{dsm['rows']:,} rows", f"timestamp: {c['timestamp_column']}", f"horizon: {c['horizon']} steps", "frequency: " + (c["frequency"] or "inferred by the engine"),
             "model: " + (", ".join(dict.fromkeys(str(s.get('selected_model')) for s in R.ok_series(r))) or "none"), "DEMO" if r.get("mode") == "demo" else "live engine"]
    st.markdown(f"<div class='ufx-repcard'><div class='ufx-eyebrow'>Report · generated {VM.safe(r['created'])}</div><h3>{VM.safe(c['target_column'])} · {VM.safe(dsm['name'])}</h3>"
                "<div class='meta'>" + "".join(f"<span>{VM.safe(x, 60)}</span>" for x in chips) + "</div></div>", unsafe_allow_html=True)
    k = st.columns(4)
    UI.kpi(k[0], "Dataset", dsm["name"] if len(dsm["name"]) <= 20 else dsm["name"][:11] + "…" + dsm["name"][-8:], f"{dsm['rows']:,} rows · {dsm['name']}")
    UI.kpi(k[1], "Target", c["target_column"], f"timestamp: {c['timestamp_column']}")
    UI.kpi(k[2], "Horizon", f"{c['horizon']} steps", "frequency: " + (c["frequency"] or "inferred by the engine"))
    UI.kpi(k[3], "Generated", r["created"], "DEMO" if r.get("mode") == "demo" else "live engine")
    if r["status"] != "ok":
        e = r.get("error") or {}
        st.error(f"This run produced no forecast, so there is nothing to download. {e.get('code', '')}: {e.get('message', '')}")
        for i in ((r.get("response") or {}).get("issues") or []) + [i for x in ((r.get("response") or {}).get("series") or []) for i in (x.get("issues") or [])]:
            st.write("• " + VM.safe_md(f"{i.get('code')}: {i.get('message')}"))
    else:
        try:
            with st.spinner("Preparing the report files…"):
                stem = R.file_stem(r)
                csv_b, html_s, obs_b = R.forecast_csv(r), R.report_html(r), R.observed_csv(r)
        except Exception as ex:                                   # noqa: BLE001
            SV.log.error("report build failed (%s) at %s", type(ex).__name__, SV.where(ex))
            st.error(f"The report files for this run could not be prepared ({type(ex).__name__}). The saved run is unchanged.")
        else:
            d = st.columns(3)
            d[0].download_button("Download forecast (CSV)", csv_b, file_name=f"{stem}.csv", mime="text/csv", key="dl_csv", type="primary", icon=":material/table:",
                                 width="stretch", on_click=_toast(f"{stem}.csv"))
            d[1].download_button("Download report (HTML)", html_s.encode("utf-8"), file_name=f"{stem}.html", mime="text/html", key="dl_html", icon=":material/article:",
                                 width="stretch", on_click=_toast(f"{stem}.html"))
            d[2].download_button("Download observed values (CSV)", obs_b, file_name=f"{stem}_observed.csv", mime="text/csv", key="dl_obs", icon=":material/timeline:",
                                 width="stretch", on_click=_toast(f"{stem}_observed.csv"))
            st.caption("The forecast CSV holds the engine's exact values: one row per step, with lower and upper bounds where the engine returned them (blank where it did not). "
                       "The observed CSV holds the stored recent values, labelled as observed.")
            t_prev, t_vals, t_chart = st.tabs(["Report preview", "Forecast file contents", "Chart"])
            with t_prev:
                st.iframe(CH.data_url(html_s), height=640)
                st.caption("This is the HTML file the download button saves, shown here unchanged.")
            with t_vals:
                st.dataframe(pd.read_csv(io.BytesIO(csv_b), float_precision="round_trip"), width="stretch", hide_index=True)
                st.caption("The rows of the forecast CSV, read back from the file contents.")
            with t_chart:
                for s in R.ok_series(r):
                    f = R.series_facts(r, s)
                    st.markdown(f"**{VM.safe_md(f['target'])}" + (f" · {VM.safe_md(f['group'])}" if f["group"] is not None else "") + "**")
                    fc = VM.forecast_frame(s)
                    st.iframe(CH.forecast_chart_src(H.tail_frame(r, f["group"]), s, f["target"], f["direction"]["label"], max(10 * len(fc), 120)), height=486)
                    body = "".join(f"<div class='ufx-kv'><span>{VM.safe(a)}</span><b>{VM.safe(b_)}</b></div>" for a, b_ in (
                        ("Selected model", f["selected_model"]), ("Why", f["selection_reason"]), ("Validation MAE", VM.fmt_num(f["val_mae"])),
                        ("Forecast range", f"{VM.fmt_num(f['forecast_min'])} to {VM.fmt_num(f['forecast_max'])}"),
                        ("Prediction intervals", "available" if f["intervals"] else f"unavailable: {f['interval_reason']}"), ("Warnings", str(len(f["issues"])))))
                    st.markdown(f"<div class='ufx-panel'><h4>Summary</h4>{body}</div>", unsafe_allow_html=True)
    b = st.columns([1.6, 1.4, 4])
    if b[0].button("Open forecast result", key="rep_open", icon=":material/show_chart:", disabled=r["status"] != "ok", width="stretch"):
        go_to_run(r)
    if b[1].button("Delete this report", key="rep_delete", icon=":material/delete:", width="stretch"):
        S["confirm_delete"] = pick
    if S.get("confirm_delete") == pick:
        st.warning("Delete this report permanently? This cannot be undone.")
        y, n, _ = st.columns([1.2, 1, 5])
        if y.button("Yes, delete it", key="rep_delete_yes", type="primary", width="stretch"):
            ok = kept.delete(pick)
            S.pop("confirm_delete", None)
            if (S.get("view_record") or {}).get("id") == pick:
                S.pop("view_record", None)
            S["_rep_flash"] = "The report was deleted." if ok else "The report could not be deleted: its file was not found."
            st.rerun()
        if n.button("Keep it", key="rep_delete_no", width="stretch"):
            S.pop("confirm_delete", None)
            st.rerun()


try:
    with st.container(key=f"ufa_page_{S.step}"):                  # each page has its own key, so moving between pages replays its entrance animation
        {"upload": screen_upload, "preview": screen_preview, "configure": screen_configure, "processing": screen_processing, "results": screen_results,
         "insights": screen_insights, "reports": screen_reports}.get(S.step, screen_upload)()
except Exception as _ex:                                          # noqa: BLE001  last line of defence: no traceback on the page, no values in the log
    SV.log.error("interface error %s on step=%s at %s", type(_ex).__name__, S.get("step"), SV.where(_ex))
    st.error(f"Something went wrong in the interface ({type(_ex).__name__}). No forecast is shown and your file was not changed. "
             "Start over, and if it happens again check the application log.")
    if st.button("Start over", key="btn_guard_restart"):
        reset_all()
