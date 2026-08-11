@echo off
setlocal
cd /d "%~dp0"

echo This compatibility launcher now starts AI-Enabled Contracts Lab.
call "%~dp0run-ai-enabled-contracts-lab.bat"
exit /b %errorlevel%
