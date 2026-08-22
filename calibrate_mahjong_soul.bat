@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Drag the Mahjong Soul 34-tile guide image onto this BAT file.
  echo Or run: calibrate_mahjong_soul.bat "C:\path\to\tile-guide.png"
  pause
  exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
  %PYTHON_CMD% -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
".venv\Scripts\python.exe" main.py --ocr-calibrate-mahjong-soul "%~1" || goto :error
echo.
echo Mahjong Soul templates are ready. You can now run start_live_viewer.bat.
pause
exit /b 0

:error
echo Calibration failed.
pause
exit /b 1
