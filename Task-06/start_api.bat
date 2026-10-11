
@echo off
cd /d "%~dp0\.."
start "ML API Server" cmd /k "python -m uvicorn app:app --app-dir Task-06 --reload"
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000/docs
echo API Docs: http://127.0.0.1:8000/docs