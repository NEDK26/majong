@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] 正在创建本地 Python 环境...
  %PYTHON_CMD% -m venv .venv || goto :error
)

echo [2/3] 正在检查并安装依赖...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

echo [3/3] 正在启动牌理镜顶部提示条...
".venv\Scripts\python.exe" main.py --live
goto :end

:error
echo.
echo 启动失败。请确认已经安装 Python 3.10 或更新版本，并勾选 Add Python to PATH。
pause

:end
endlocal
