@echo off
setlocal

set "ROOT=%~dp0.."
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "DEPS_MARK=%ROOT%\.deps_ok"

if not exist "%PY%" (
  echo [setup] Creating virtual environment...
  py -3.10 -m venv "%VENV%" || py -3 -m venv "%VENV%" || python -m venv "%VENV%"
)

if not exist "%DEPS_MARK%" (
  echo [setup] Installing dependencies (first time only)...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r "%ROOT%\requirements.txt"
  if errorlevel 1 (
    echo [error] Dependency install failed.
    pause
    exit /b 1
  )
  echo ok>"%DEPS_MARK%"
)

"%PY%" "%~dp0Main.py"
pause
