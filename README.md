# AETHERCAST

Upload a CSV with dates and numbers, check it, configure a forecast, and get the forecast with the evidence behind it:
how each candidate model did on a validation period inside your data, prediction intervals where the engine can
estimate them, warnings, limitations and a plain-language explanation written by fixed rules (no language model).

This repository holds only what the web app needs: the interface (`universal_app/`) and the forecasting engine and
explanation layer it calls (`src/universal_forecasting/`). Research data, experiments and reports are not included.

## Run it

- Hosted (Streamlit Community Cloud): entry point `streamlit_app.py`, Python 3.12, packages in `requirements.txt`.
- Locally: `pip install -r requirements.txt`, then `streamlit run streamlit_app.py`
  (or `python universal_app/launch.py` for the local, non-hosted settings on http://localhost:8503).

## What the hosted version does differently

- Smaller limits for the small shared server: 15 MB, 250,000 rows, 200 columns per file.
- Your file is sent to the server, kept in memory for your session and never written to disk.
- Saved runs (Insights, Forecast Reports) last only for your browser session. They are lost when the tab is closed or
  reloaded, after a long pause, or when the app restarts or sleeps. Download the CSV and HTML files to keep a run.
- A forecast cannot be cancelled and has no time limit; large files can take a minute.

## Limits

There is no sign-in, rate limiting or per-user isolation beyond Streamlit's own sessions, and the app has not been
security-tested for public use. Forecasts are estimates, validation figures are not test results, and nothing here
is financial advice.
