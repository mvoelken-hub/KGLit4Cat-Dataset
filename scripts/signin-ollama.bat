@echo off
setlocal

set "OLLAMA_CONTAINER="
for /f "delims=" %%i in ('docker ps --filter "label=com.docker.compose.service=ollama" --format "{{.ID}}"') do (
    set "OLLAMA_CONTAINER=%%i"
    goto found_container
)

echo No running Ollama Docker container was found.
echo Start the stack first with up-cpu.bat or up-gpu.bat, then run this script again.
exit /b 1

:found_container
echo Running Ollama sign-in inside Docker container %OLLAMA_CONTAINER%...
echo.
echo If Ollama prints a connect link, open it in your browser and complete the sign-in.
echo.
docker exec "%OLLAMA_CONTAINER%" ollama signin

endlocal
