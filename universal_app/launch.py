r"""Launch the Universal Forecasting interface on port 8503 with its monochrome theme, without touching the other apps'
configuration (the theme is passed as command-line options, not written to a shared .streamlit/config.toml).
The server listens on this computer only (localhost).

    .venv\Scripts\python.exe universal_app\launch.py           normal mode (default): the forecasting engine and the rule-based explanations
    .venv\Scripts\python.exe universal_app\launch.py --demo    demo mode (stored result for the bundled synthetic sample)
    set UFA_SERVICE=not_connected, then the first command     engine switched off (validation only)

Completed forecast runs are saved as one small JSON file each in  <project>\universal_app_data\runs  (set UFA_DATA_DIR
to use another folder). Uploaded files themselves are never saved.
"""
import sys
from pathlib import Path

from streamlit.web import cli as stcli

if __name__ == "__main__":
    import os
    if "--demo" in sys.argv:                     # same as setting UFA_SERVICE=demo
        sys.argv.remove("--demo")
        os.environ["UFA_SERVICE"] = "demo"
    os.environ.setdefault("UFA_STAGE_PAUSE", "0.45")   # seconds each finished stage stays readable in the progress panel (display only)
    os.environ.setdefault("UFA_DATA_DIR", str(Path(__file__).resolve().parents[1] / "universal_app_data" / "runs"))      # run history for Insights and Reports
    app = str(Path(__file__).resolve().parent / "app.py")
    sys.argv = ["streamlit", "run", app, "--server.port", "8503", "--server.address", "localhost", "--server.headless", "true", "--browser.gatherUsageStats", "false",
                "--server.maxUploadSize", "50",
                "--server.enableXsrfProtection", "true", "--server.enableCORS", "true",      # refuse cross-site requests (Streamlit defaults, stated explicitly)
                "--client.showErrorDetails", "none", "--client.toolbarMode", "minimal",      # no tracebacks on the page; no Deploy / developer menu
                "--theme.base", "dark", "--theme.primaryColor", "#e5e5e5", "--theme.backgroundColor", "#000000",
                "--theme.secondaryBackgroundColor", "#141415", "--theme.textColor", "#f5f5f5", "--theme.font", "sans serif"] + sys.argv[1:]
    sys.exit(stcli.main())
