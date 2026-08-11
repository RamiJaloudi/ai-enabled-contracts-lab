$ErrorActionPreference = "Stop"

Write-Host "Creating the AI-Enabled Contracts Lab Python 3.12 environment..." -ForegroundColor Cyan
uv sync --python 3.12 --extra dev --extra cu129

Write-Host "Checking the environment..." -ForegroundColor Cyan
uv run --extra cu129 contractlab doctor

Write-Host "AI-Enabled Contracts Lab is ready. Double-click run-ai-enabled-contracts-lab.bat or run .\run.ps1" -ForegroundColor Green
