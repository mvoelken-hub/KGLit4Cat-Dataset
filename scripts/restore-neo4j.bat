@echo off
setlocal
cd /d "%~dp0\.."

if "%~1"=="" (
    echo Usage:
    echo   scripts\restore-neo4j.bat backups\neo4j\backup-file.cypher
    exit /b 1
)

set "BACKUP_FILE=%~1"

if not exist "%BACKUP_FILE%" (
    echo Backup file not found: %BACKUP_FILE%
    exit /b 1
)

if not exist .env.development (
    echo Missing .env.development. Create it from .env.development.example first.
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r "^NEO4J_USER= ^NEO4J_PASSWORD=" .env.development`) do set "%%A=%%B"

for /f %%I in ('docker compose --env-file .env.development -f docker-compose.dev.yml ps -q neo4j') do set "NEO4J_CONTAINER=%%I"

echo Copying backup into Neo4j container...
docker cp "%BACKUP_FILE%" "%NEO4J_CONTAINER%:/tmp/restore.cypher"

if errorlevel 1 (
    echo Failed to copy backup file into Neo4j container.
    exit /b 1
)

echo Restoring Neo4j graph from %BACKUP_FILE% ...
docker compose --env-file .env.development -f docker-compose.dev.yml exec -T neo4j cypher-shell -u "%NEO4J_USER%" -p "%NEO4J_PASSWORD%" -f /tmp/restore.cypher

if errorlevel 1 (
    echo Restore failed.
    exit /b 1
)

echo Restore completed successfully.
endlocal
exit /b 0
