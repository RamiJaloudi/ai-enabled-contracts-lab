@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://localhost:8501"

rem If the app is already healthy, open it instead of starting a duplicate.
powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing '%APP_URL%/_stcore/health' -TimeoutSec 2; if ($response.StatusCode -eq 200) { exit 0 } } catch { exit 1 }; exit 1" >nul 2>nul
if not errorlevel 1 (
    echo AI-Enabled Contracts Lab is already running.
    echo Opening %APP_URL%
    start "" "%APP_URL%"
    exit /b 0
)

where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is not available on PATH.
    echo Install uv, then run setup.ps1 before trying again.
    pause
    exit /b 1
)

echo Starting AI-Enabled Contracts Lab...
echo URL: %APP_URL%
echo Keep this window open while using the application.
echo Press Ctrl+C here to stop AI-Enabled Contracts Lab.
echo.

uv run --extra cu129 streamlit run streamlit_app.py --server.address=localhost --server.port=8501

if errorlevel 1 (
    echo.
    echo AI-Enabled Contracts Lab stopped with an error.
    pause
    exit /b 1
)

endlocal
