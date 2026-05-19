@echo off
setlocal
cd /d "%~dp0"

echo Select startup mode:
echo   1. Production CPU  - API, Neo4j, and Ollama run in Docker
echo   2. Development CPU - Neo4j and Ollama run in Docker, API and frontend run locally
set /p MODE="Enter 1 or 2: "

if "%MODE%"=="1" goto production
if "%MODE%"=="2" goto development

echo Invalid selection.
exit /b 1

:production
if not exist .env.production (
    copy .env.production.example .env.production >nul
    echo Created .env.production from .env.production.example
)

docker compose --env-file .env.production up -d --build
call :wait_for_api
goto open_browser

:development
if not exist .env.development (
    copy .env.development.example .env.development >nul
    echo Created .env.development from .env.development.example
)

docker compose --env-file .env.development -f docker-compose.dev.yml stop frontend >nul 2>nul
docker compose --env-file .env.development -f docker-compose.dev.yml up -d --build neo4j ollama
start "API" /D "%~dp0backend" powershell -NoExit -ExecutionPolicy Bypass -Command "uv run --env-file ../.env.development uvicorn app.main:fastapi_app --host 127.0.0.1 --port 8000 --reload"
call :start_frontend
if errorlevel 1 exit /b 1
call :wait_for_api
call :wait_for_frontend
goto open_browser

:start_frontend
echo Starting frontend dev server...
if exist "C:\Program Files\nodejs\node.exe" set "PATH=C:\Program Files\nodejs;%PATH%"
where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Install Node.js or add npm to PATH, then rerun this script.
    exit /b 1
)
if not exist "%~dp0frontend\node_modules" (
    start "Frontend" /D "%~dp0frontend" cmd /k "npm install && npm run dev -- --host 127.0.0.1"
) else (
    start "Frontend" /D "%~dp0frontend" cmd /k "npm run dev -- --host 127.0.0.1"
)
exit /b 0

:wait_for_api
echo Wait for API startup...
for /l %%i in (1,1,60) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/docs' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }"
    if not errorlevel 1 (
        echo API is reachable.
        exit /b 0
    )
    echo API not reachable yet. Retry %%i/60...
    timeout /t 2 /nobreak >nul
)
echo API was not reachable after 120 seconds. Opening URLs anyway...
exit /b 0

:wait_for_frontend
echo Wait for frontend startup...
for /l %%i in (1,1,60) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3000/' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }"
    if not errorlevel 1 (
        echo Frontend is reachable.
        exit /b 0
    )
    echo Frontend not reachable yet. Retry %%i/60...
    timeout /t 2 /nobreak >nul
)
echo Frontend was not reachable after 120 seconds. Opening URLs anyway...
exit /b 0

:open_browser
echo Opening browser...
start "SIMONE Frontend" "http://127.0.0.1:3000/"
start "Neo4j Browser" "http://127.0.0.1:7474/browser/"
start "API Docs" "http://127.0.0.1:8000/docs"
endlocal
exit /b 0
