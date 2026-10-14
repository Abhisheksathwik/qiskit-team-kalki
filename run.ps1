$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv/Scripts/python.exe")) {
    py -3.11 -m venv .venv
}
$python = ".venv/Scripts/python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
& $python -m streamlit run app.py
