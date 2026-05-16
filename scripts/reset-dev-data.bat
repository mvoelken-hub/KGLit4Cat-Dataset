@echo off
setlocal
cd /d "%~dp0\.."

echo WARNING: This will reset the local Neo4j database.
set /p BACKUP_CHOICE="Create backup before reset? (Y/N): "

if /I "%BACKUP_CHOICE%"=="Y" (
    call scripts\backup-neo4j.bat
)

echo Stopping Neo4j development container...
docker compose --env-file .env.development -f docker-compose.dev.yml down neo4j

echo Removing Neo4j data directory...
if exist data\docker\neo4j\data (
    rmdir /s /q data\docker\neo4j\data
)

echo Development Neo4j database reset complete.
echo The next startup will initialize Neo4j with the current .env.development credentials.

endlocal
exit /b 0
