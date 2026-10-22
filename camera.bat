@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Main app Python not found: venv\Scripts\python.exe
  pause
  exit /b 1
)
venv\Scripts\python.exe run.py --cfg configs/onnx_infer.yaml --realtime --dri_video 0 --require-nvidia %*
if errorlevel 1 pause
