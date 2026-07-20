@echo off

if not exists "%~dp0venv\" (
    echo "venv not found, creating..."
    python -m venv venv
) else (
    echo "Found venv"
)


"./venv/Scripts/python.exe" -m pip install --upgrade pip
"./venv/Scripts/pip.exe" install -r requirements.txt

"./venv/Scripts/python.exe" installer.py

pause