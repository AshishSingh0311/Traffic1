@echo off
REM Activate virtualenv if exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
streamlit run streamlit_app.py --server.address localhost --server.port 5000
pause
