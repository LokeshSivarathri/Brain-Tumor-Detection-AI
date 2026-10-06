@echo off
echo ===================================================
echo Starting Brain Tumor Detection AI Web Application
echo ===================================================
echo Opening at http://127.0.0.1:8080/
echo.

if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe backend\app.py
) else if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe backend\app.py
) else (
    python backend\app.py
)
pause
