@echo off
REM TrialMatch-Lite launcher: double-click this file, then use the browser window that opens.
cd /d "%~dp0"
echo Starting TrialMatch-Lite on http://127.0.0.1:8000 (first start loads the models, ~20 s) ...
start "" /b cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8000"
python -m app.server
pause
