@echo off
chcp 65001 >nul
setlocal
title RAG Enterprise Assistant

cd /d "%~dp0"

set "PYTHON=S:\PythonAILearningCourse\6.Transformers\.venv\Scripts\python.exe"

echo ============================================
echo   RAG Enterprise Assistant
echo ============================================
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python venv not found:
    echo         %PYTHON%
    echo.
    echo Please update PYTHON in start.bat.
    pause
    exit /b 1
)

if "%dsAPIkey%"=="" (
    echo [WARN] dsAPIkey is not set in this window.
    echo        If authentication fails, reopen the terminal or check the system environment variable.
    echo.
)

if not exist "data\index\faiss.index" (
    echo [INFO] Index not found, building...
    "%PYTHON%" scripts\build_index.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to build index.
        pause
        exit /b 1
    )
    echo.
)

echo [INFO] Starting server. First model load takes about 20 seconds...
echo [INFO] Browser will open http://127.0.0.1:7860
echo [INFO] Press Ctrl+C or close this window to stop.
echo.

"%PYTHON%" app\ui.py

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited unexpectedly. Check the output above or logs\app.log
    pause
)

endlocal
