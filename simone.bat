@echo off
REM SIMONE CLI wrapper — delegates to the backend virtual environment
cd /d "%~dp0backend"
uv run simone %*
