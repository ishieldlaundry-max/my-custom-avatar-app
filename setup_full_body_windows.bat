@echo off
setlocal

REM Official MimicMotion + DWPose setup for the optional offline full-body mode.
REM Keep this in a separate environment from the FasterLivePortrait venv.

set "ROOT=%~dp0third_party\MimicMotion"
if not exist "%ROOT%\inference.py" (
    git clone --depth 1 https://github.com/Tencent/MimicMotion.git "%ROOT%"
    if errorlevel 1 exit /b 1
)

if not exist "%ROOT%\.venv\Scripts\python.exe" (
    py -3.11 -m venv "%ROOT%\.venv"
    if errorlevel 1 (
        echo Python 3.11 is required for the MimicMotion environment.
        exit /b 1
    )
)

call "%ROOT%\.venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install diffusers==0.27.0 transformers==4.32.1 huggingface_hub==0.25.2 numpy==1.26.4 matplotlib decord einops omegaconf onnxruntime-gpu

if not exist "%ROOT%\models\DWPose" mkdir "%ROOT%\models\DWPose"
if not exist "%ROOT%\models\DWPose\yolox_l.onnx" (
    curl.exe -L "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true" -o "%ROOT%\models\DWPose\yolox_l.onnx"
)
if not exist "%ROOT%\models\DWPose\dw-ll_ucoco_384.onnx" (
    curl.exe -L "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true" -o "%ROOT%\models\DWPose\dw-ll_ucoco_384.onnx"
)
if not exist "%ROOT%\models\MimicMotion_1-1.pth" (
    curl.exe -L "https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth?download=true" -o "%ROOT%\models\MimicMotion_1-1.pth"
)

echo.
echo Full-body backend installed.
echo The Stable Video Diffusion base model downloads on first render.
echo Use full_body.py with a full-body target image and recorded webcam video.
endlocal