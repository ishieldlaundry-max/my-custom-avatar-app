@echo off
setlocal
cd /d "%~dp0"
REM Separate build environment: never change torch/onnx/SF3D dependencies here.
if not exist "venv\Scripts\python.exe" (
  echo Missing main app environment: venv\Scripts\python.exe
  exit /b 1
)
if not exist ".desktop-build-env\Scripts\python.exe" (
  venv\Scripts\python.exe -m venv .desktop-build-env
  if errorlevel 1 exit /b 1
)
.desktop-build-env\Scripts\python.exe -m pip install -r requirements_desktop.txt
if errorlevel 1 exit /b 1
.desktop-build-env\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name AvatarDesktop --distpath generated-artifacts\desktop --workpath generated-artifacts\desktop-work --specpath generated-artifacts desktop_launcher.py
if errorlevel 1 exit /b 1
copy /Y generated-artifacts\desktop\AvatarDesktop.exe AvatarDesktop.exe
echo Built AvatarDesktop.exe in the project folder. This launcher requires the existing venv and model files.
pause
