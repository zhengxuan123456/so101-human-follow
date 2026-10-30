@echo off
setlocal
chcp 65001 >nul

set "SO101_PROJECT=%~dp0"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "SO101_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined SO101_PYTHON (
  where python >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] 未找到 Python。请先激活安装了依赖的 Conda 环境。
    pause
    exit /b 1
  )
  set "SO101_PYTHON=python"
)

if not exist "%SO101_PROJECT%config.json" (
  echo [ERROR] 缺少 config.json。
  echo 请先复制 config.example.json 为 config.json，并填写本机端口与机器人 ID。
  pause
  exit /b 1
)

set "PYTHONPATH=%SO101_PROJECT%\src"
pushd "%SO101_PROJECT%"
echo 正在启动仅预览模式：http://127.0.0.1:7860/
echo 此模式不会连接 COM4，也不会控制舵机。
"%SO101_PYTHON%" -u -B -m so101_human_follow.web_app --config "%SO101_PROJECT%\config.json" --http-port 7860
popd
pause
