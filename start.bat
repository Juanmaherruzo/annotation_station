@echo off
REM Launch annotation-station: FastAPI backend + Vite frontend, then open the UI.
REM Paths are resolved relative to this script, so the repository can live
REM anywhere. Override VENV or NPM below only if your setup differs from the
REM one described in INSTALL.md.

setlocal
set ROOT=%~dp0
set BACKEND=%ROOT%backend
set FRONTEND=%ROOT%frontend

REM Python virtual environment. INSTALL.md creates this at <repo>\.venv.
if "%VENV%"=="" set VENV=%ROOT%.venv
set UVICORN=%VENV%\Scripts\uvicorn.exe

REM Node comes from PATH unless NPM is set explicitly.
if "%NPM%"=="" set NPM=npm.cmd

if not exist "%UVICORN%" (
    echo.
    echo ERROR: uvicorn not found at "%UVICORN%".
    echo Create the environment and install the backend first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     cd backend ^&^& pip install -e ".[dev]"
    echo See INSTALL.md for the full sequence.
    echo.
    pause
    exit /b 1
)

where %NPM% >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: npm not found on PATH. Install Node.js 18 or newer,
    echo or set NPM to its full path before running this script.
    echo.
    pause
    exit /b 1
)

echo Starting annotation-station...

start "annotation-station - Backend" cmd /k "set KMP_DUPLICATE_LIB_OK=TRUE && cd /d %BACKEND% && "%UVICORN%" app.main:app --host 127.0.0.1 --port 8000 --reload"

start "annotation-station - Frontend" cmd /k "cd /d %FRONTEND% && %NPM% run dev"

echo Waiting for the backend to load SAM (this takes 10-30 s on first start)...
:wait_backend
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 goto wait_backend

echo Waiting for the frontend...
:wait_frontend
timeout /t 2 /nobreak >nul
curl -s http://localhost:5173 >nul 2>&1
if errorlevel 1 goto wait_frontend

echo Ready. Opening the browser...
start http://localhost:5173
endlocal
