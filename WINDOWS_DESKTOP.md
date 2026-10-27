# Windows desktop launcher and NVIDIA policy

## What this delivers

AvatarDesktop is a native Windows window hosting the local Gradio interface in
Microsoft Edge WebView2. It does not open a browser tab. The server listens only
on 127.0.0.1. It is a launcher, not a portable bundle of Python, models, or CUDA.
Keep it in the existing FasterLivePortrait-local folder.

Packaging does not improve model quality or inference speed by itself.
The single-photo live path is experimental. Calibrated head turns are bounded
to protect likeness. Shoulder movement is a local 2-D image deformation, and
directional shading is illustrative, not reconstructed 3-D lighting or shadows.
Packaging alone does not improve motion accuracy.

## NVIDIA requirement

Windows ONNX model sessions require CUDAExecutionProvider and explicitly disable
CPU execution-provider fallback. Live capture also checks CUDA PyTorch and makes
a real GPU allocation. If CUDA libraries or supported GPU kernels are missing,
startup fails with an error instead of running the portrait models on Intel/CPU.
CPU preprocessing, UI, camera capture and file IO still use CPU normally.
The Linux Replit preview can use CPU; --require-nvidia enables the strict policy
explicitly there too.

The previous Windows log showed CPUExecutionProvider. That installation needs
a compatible CUDA PyTorch / onnxruntime-gpu / cuDNN combination before it will
pass this check. Do not reinstall all requirements or modify the SF3D environment.
Collect main-environment versions and available providers before choosing GPU
packages. A driver alone is not sufficient; ORT must load its CUDA libraries.
Some ONNX graphs may be rejected by the strict no-CPU-operators policy.

### Volumetric warping compatibility

The portrait warping checkpoint uses two 5-D GridSample operations. ONNX Runtime
CUDA can load this model but fail on the first frame with "Only 4-D tensor is
supported". The CUDA warping adapter splits the existing checkpoint in memory,
runs those sampling operations in PyTorch CUDA, and runs the other stages in
ONNX CUDA with CPU fallback disabled. Intermediate tensors stay on CUDA using
explicit I/O binding and stream synchronization. No new model downloads,
TensorRT plugins, package reinstalls, or executable rebuild are needed.

The startup log identifies this path as "ONNX CUDA + PyTorch CUDA GridSample".
CPU numerical comparison validates the graph partition and sampling semantics;
it does not establish Windows CUDA speed, memory usage, or successful rendering.
On NVIDIA hardware, run the real-checkpoint CUDA regression with:

```bat
venv\Scripts\python.exe -m unittest discover -s tests -p test_cuda_warping.py -v
```

The CUDA regression is skipped when NVIDIA hardware or the checkpoint is absent.

Windows, not the app, selects Intel versus NVIDIA for WebView/browser graphics.
In Windows Settings > System > Display > Graphics, select AvatarDesktop.exe
and choose High performance (NVIDIA). Use the same preference for your browser
if using the browser viewer. Verify actual GPU engine usage in Task Manager;
do not assume a MediaPipe "GPU" label identifies the NVIDIA adapter.

## Build the executable on Windows

Prerequisites: the existing working main venv, model files, NVIDIA driver,
and Microsoft Edge WebView2 Runtime. Build on the target Windows computer.
Linux cannot produce a tested Windows PyInstaller executable.

In CMD:

```bat
cd /d C:\Users\emman\FasterLivePortrait-local
build_desktop_windows.bat
```

The script creates a separate build-only environment and installs
requirements_desktop.txt there. It does not modify CUDA PyTorch, ONNX Runtime,
or Stable Fast 3D. The resulting AvatarDesktop.exe is copied into the project
folder. Double-click it. The first model load can take several minutes.
This is an unsigned executable; sign it before distributing publicly.

The app window can be resized down to 640 x 480. Smaller layouts use scrolling.
Minimizing to the taskbar hides the app; it does not resize its content or stop
tracking. Use Stop live tracking or Exit to release the webcam.

## Continuous webcam portrait

Use the native App > Start live portrait from photo menu, choose a portrait,
and confirm closing the workspace server to free its GPU memory. The live
portrait opens in a separate OpenCV window. This is continuous capture, unlike
the workspace's recorded-video input. Close other webcam applications first.
Live webcam mode also opens a **Your camera** window showing the same input
frames that drive inference (mirrored for display only). Look straight ahead
with a relaxed mouth and open eyes, then press **C** in either OpenCV window
to calibrate. The portrait stays still until calibration. Press **C** again
to reset the expression reference, or **Q** / Escape to stop. Closing either
preview window also stops tracking. Face detection is refreshed on every live
frame; when no face is detected, the portrait pauses and the camera status says
so. Preview updates are paced by inference, not a promise of 30 FPS.

### Head, shoulders and shading

The **Your camera** window has a clickable toolbar below the camera image:
**Calibrate**, **Head ON/OFF**, **Set shoulders**, **Shoulders off**,
**Shading ON/OFF**, and **Stop tracking**. These perform the same actions as
the keyboard shortcuts below. Toolbar clicks do not select shoulder points.
Controls are processed between inference frames, so a slow render can delay a click.

Live webcam mode uses adaptive expression/pose smoothing. Larger changes pass
through faster than small tremors; this is not a measured accuracy guarantee.
Head motion starts enabled: **H** toggles it. Relative yaw is limited to 15 degrees,
pitch/roll to 10 degrees. Target scale, translation and expression bounds remain
protected; offline rendering retains its existing pose policy.

For shoulder motion, keep your upper body visible and press **S**. Click the two
shoulders in **Your camera**, then the corresponding two shoulders in **Render**.
Yellow markers confirm the selections. Stay still while selecting and press **C**
to calibrate face and shoulders together. Use textured clothing near the shoulder
points; optical flow cannot reliably track a plain, featureless area. The shoulder
effect is bounded and fades out above the selected shoulder region. **B** turns
it off; **S** starts a new selection. Lost tracking freezes the last shoulder
offset and asks for recalibration, rather than inventing movement.

**L** toggles subtle face-local simulated shading driven by calibrated head
orientation. It starts enabled. It does not estimate room lighting, reconstruct
depth, or create physically correct cast shadows on the background. Shoulder
flow, warping and shading are conventional CPU image processing; all neural
inference remains under the NVIDIA-only Windows policy. These effects may add
latency; Windows frame rate and visual fidelity need real-device evaluation.

Use App > Stop live tracking to stop it. Restart AvatarDesktop to return to
the workspace after live mode. Closing AvatarDesktop stops its child processes.

## Native skeletal tracking

Use **App > Start native skeletal tracking** to run a separate OpenCV webcam
mode without choosing a portrait. It closes the workspace server to free GPU
memory, then opens **Your camera — skeletal mode** and **Approximate 3D
skeleton**. Press **C** while your full body is visible in a neutral pose to
save the body proportions and expression baseline. Press **Q** or Escape, or
close either preview window, to stop tracking.

This mode uses the existing DWPose Wholebody ONNX assets provisioned by
`setup_full_body_windows.bat`. Inference is required to use
`CUDAExecutionProvider` from the main application environment; it does not
fall back to CPU. The MimicMotion source and model files must be present under
`third_party\MimicMotion`, and the main environment must have a compatible
`onnxruntime-gpu` installation. Missing assets or CUDA support stop the mode
with an explicit error.

DWPose supplies 2D body, hand, and face landmarks. The skeletal frame stores
body-height-relative coordinates, confidence, 2D-derived bone rotations, and
heuristic eye/mouth coefficients. Its Z translation is an approximate body-scale
estimate from apparent subject size; it is not measured or metric depth, and
the face coefficients are not model-predicted blendshapes. The stick figure
therefore shows an approximate monocular 2.5D pose, not a physically calibrated
3D reconstruction.

Capture and inference run independently with a one-frame queue, so slow
inference drops stale frames rather than holding up camera preview. The
workspace's photo-face-swap mode remains a separate path.

For a direct launch from the existing application environment:

```bat
venv\Scripts\python.exe run.py --skeletal_tracking --dri_video 0 --require-nvidia
```

Alternatively, no executable build is needed to start the same live path:

```bat
camera.bat --src_image "C:\path\portrait.png" --paste_back
```

## Errors and logs

App > Open logs opens the local logs folder. Startup and live runs have separate
timestamped logs. If startup fails, inspect desktop-*.log; if live mode fails,
inspect live-*.log or skeletal-*.log. Logs are local, not uploaded automatically.

This launcher has CPU unit tests here, but the Windows executable, WebView2
camera permissions, NVIDIA inference, and end-to-end live motion must still be
verified on Windows. No successful hardware test is implied by building it.
