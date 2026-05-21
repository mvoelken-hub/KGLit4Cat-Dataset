@echo off
REM SIMONE CLI wrapper - delegates to the backend virtual environment
uv --directory "%~dp0backend" run simone %*
