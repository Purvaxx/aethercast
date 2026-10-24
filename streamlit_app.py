"""AetherCast AI hosted entry point (Streamlit Community Cloud runs this file).

It runs the same interface as `universal_app/launch.py` does locally, with three hosted settings:
  * UFA_HOSTED=1     smaller upload limits for the small shared server, wording that says the file goes to the server,
                     and saved runs kept per browser session only (never in a shared folder other visitors could see);
  * UFA_STAGE_PAUSE  each finished progress stage stays readable for a moment (display only, as locally);
  * UFA_DATA_DIR     removed, so run history is never written to the server's disk.
Server and theme options live in .streamlit/config.toml. The forecasting engine is unchanged.
"""
import os
import runpy
from pathlib import Path

os.environ["UFA_HOSTED"] = "1"
os.environ.setdefault("UFA_STAGE_PAUSE", "0.45")
os.environ.pop("UFA_DATA_DIR", None)
runpy.run_path(str(Path(__file__).resolve().parent / "universal_app" / "app.py"), run_name="__main__")
