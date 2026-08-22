@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating local Python environment...
  %PYTHON_CMD% -m venv .venv || goto :error
)

echo [2/3] Checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

echo [3/3] Starting Mahjong desktop study client...
".venv\Scripts\python.exe" main.py --live
goto :end

:error
echo.
echo Startup failed. Confirm Python 3.10 or newer is installed.
pause

:end
endlocal
