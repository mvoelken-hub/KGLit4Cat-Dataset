@echo off
setlocal
cd /d "%~dp0\.."

if not exist .env.development (
    echo Missing .env.development. Create it from .env.development.example first.
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r "^NEO4J_USER= ^NEO4J_PASSWORD=" .env.development`) do set "%%A=%%B"

if not exist .backups\neo4j mkdir .backups\neo4j

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "BACKUP_TS=%%I"
set "BACKUP_NAME=%BACKUP_TS%.cypher"
set "BACKUP_FILE=.backups\neo4j\%BACKUP_NAME%"

echo Exporting Neo4j graph to %BACKUP_FILE% ...

docker compose --env-file .env.development -f docker-compose.dev.yml exec -T neo4j cypher-shell -u "%NEO4J_USER%" -p "%NEO4J_PASSWORD%" "CALL apoc.export.cypher.all('%BACKUP_NAME%', {format: 'cypher-shell'});"

if errorlevel 1 (
    echo Backup failed.
    exit /b 1
)

for /f %%I in ('docker compose --env-file .env.development -f docker-compose.dev.yml ps -q neo4j') do set "NEO4J_CONTAINER=%%I"

docker cp "%NEO4J_CONTAINER%:/var/lib/neo4j/import/%BACKUP_NAME%" "%BACKUP_FILE%"

if errorlevel 1 (
    echo Backup export succeeded, but copying the backup file failed.
    exit /b 1
)

echo Backup created: %BACKUP_FILE%
endlocal
exit /b 0
