@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo 请把雀魂 34 种牌总览图拖到这个 BAT 文件上。
  echo 也可以运行：calibrate_mahjong_soul.bat "C:\图片路径\麻将牌.png"
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
echo 雀魂牌面模板已经准备完成，现在可以运行 start_live_viewer.bat。
pause
exit /b 0

:error
echo 校准失败。请确认图片是包含 34 种牌的雀魂“麻将牌”总览图。
pause
exit /b 1
