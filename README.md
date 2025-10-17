# TrafficFlowVision

AI-powered traffic management system combining CV, ML, fog computing and a Streamlit dashboard.

Quick start

Prereqs
- Python 3.10+ (recommended)
- pip

Install

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Run

```powershell
# run Streamlit on localhost:5000
streamlit run streamlit_app.py --server.address localhost --server.port 5000
```

Notes
- The `.streamlit/config.toml` file may contain address=0.0.0.0; use localhost when opening in a browser.
- The `attached_assets` folder is ignored by default in `.gitignore` to avoid committing large datasets.
