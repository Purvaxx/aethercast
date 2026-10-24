# AetherCast AI — Universal Time-Series Forecasting

AetherCast AI forecasts any regularly spaced time series from a single CSV file. Upload a file with a time column and
a numeric column; the app checks the file, works out sensible settings from its values, compares several forecasting
models on a validation period inside your own history, and returns the forecast with the evidence behind it.

## Features

- **Upload and validation**: strict CSV reading (nothing is filled, dropped or resampled); unclear dates, ragged rows
  and unusable columns are refused with a reason.
- **Dataset preview**: the first rows with each column's detected role, column profiles and a map of missing values.
- **Automatic configuration**: timestamp, target, frequency, horizon and series identifier are inferred from the
  data; every setting can be changed, and a confirmation is asked only when the data allows more than one reading.
- **Forecasting engine**: naive, seasonal naive, drift, exponential smoothing and ridge models, compared on the same
  validation origins; the simplest adequate model is kept. Prediction intervals come from validation errors and are
  omitted when there is too little history to estimate them.
- **Results**: an interactive chart (observed history, forecast, interval bands, exact values on hover), model
  comparison, warnings, limitations and a plain-language explanation written by fixed rules from the engine output.
- **Insights and reports**: run history with filters and charts, and downloadable CSV and HTML reports.

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Python 3.12. `python universal_app/launch.py` starts the same interface with the local settings on
http://localhost:8503 (larger upload limits, run history saved to a local folder).

## Hosted version

The hosted app (Streamlit Community Cloud) accepts files up to 15 MB, 250,000 rows and 200 columns. An uploaded file
is sent to the server, kept in memory for the session and never written to disk. Run history lasts only for the
browser session; download the CSV or HTML report to keep a result.

## Project structure

```
streamlit_app.py              hosted entry point
universal_app/                interface (pages, charts, data checks, reports)
src/universal_forecasting/    forecasting engine and explanation layer
.streamlit/config.toml        server and theme settings
```

## Limitations

There is no sign-in or rate limiting, a forecast cannot be cancelled, and the app has not been security-tested for
heavy public use. Forecasts are estimates; validation figures describe one past period of your data, are not test
results, and nothing here is financial advice.

## Author

[Purvaxx](https://github.com/Purvaxx)
