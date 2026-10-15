# Local photo-to-3D preview on Windows

This optional workflow makes a textured GLB from one image, then can bind it
to a basic humanoid skeleton locally with Blender. It does not create a
production-accurate face or body. The unseen sides and body are inferred by the
model, and the automatic rig fit is approximate. Stable Fast 3D's Windows
support is experimental.

## Before setup

1. The model is gated. Request access to
   [Stable Fast 3D on Hugging Face](https://huggingface.co/stabilityai/stable-fast-3d)
   and accept its model terms.
2. Install the Visual Studio 2022 C++ build tools required by Stable Fast 3D.
3. Keep at least 6 GB of free GPU memory available. The current app's default
   image-to-mesh pass is about 6 GB; an 8 GB card is tight, especially if a game,
   browser tracker, or other CUDA application is active.
4. The app deliberately uses a separate Python environment under
   `third_party/stable-fast-3d/.venv` so its pinned dependencies do not replace
   the existing TensorRT environment.
5. Install the free Blender desktop app from
   <https://www.blender.org/download/>. Blender is used only for local mesh
   import, automatic weights, and GLB export; it does not upload the photo.
   If Blender is installed in a custom location, set `BLENDER_EXECUTABLE` to
   its `blender.exe` path before starting the app.

## Install Stable Fast 3D

From the project folder in PowerShell:

```powershell
git clone https://github.com/Stability-AI/stable-fast-3d.git third_party/stable-fast-3d
Set-Location third_party/stable-fast-3d
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools==69.5.1 wheel
```

Install a CUDA-enabled PyTorch build compatible with the installed NVIDIA
driver, then follow Stable Fast 3D's official Windows instructions for its
remaining requirements and local CUDA extensions:

<https://github.com/Stability-AI/stable-fast-3d#installation>

Windows support is experimental. Do not install these packages into the app's
existing `venv`. After installation, use the Hugging Face CLI inside the
isolated environment to sign in with the token for the model access you
requested. Never put that token in this repository or in chat.

## Generate and rig a preview

Restart the app, expand **Live 3D avatar (prototype)**, and use **Local
photo-to-3D preview**. Use an upright, full-body image with one person, visible
arms and legs, and some space between the arms and torso. Close the webcam
tracker, games, and other GPU-heavy software first. Run once at **512** and
note the elapsed time and peak model-allocated VRAM shown below the preview.
If 512 completes, run again at **768** and compare. Do not try 768 if 512
reports CUDA out-of-memory.

When the mesh is generated, click **Rig generated GLB locally with Blender**.
The app writes a second file ending in `_rigged.glb`; it keeps the original
textured mesh intact. This basic fit supports head pose and body tracking in
the live avatar page. It does not create facial expression blendshapes, and
photo geometry may not bend cleanly at joints. The original static GLB remains
available if rigging fails.

The reported peak is PyTorch memory allocated by the model process; it is not
the GPU's total usage across all applications. The workflow logs and the
Windows Task Manager GPU memory graph can help identify memory used elsewhere.

A single close-up photo cannot provide accurate side, back, or full-body
details. This path does not add realistic facial expressions or recover
occluded geometry.
