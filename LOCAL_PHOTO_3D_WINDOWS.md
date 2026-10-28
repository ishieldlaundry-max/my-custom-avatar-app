# Local photo-to-3D preview on Windows

This optional workflow makes a textured GLB from one image, then automatically
binds it to a basic humanoid skeleton locally with Blender. The app handles both
steps; you do not need to manually rig or reload the model. It does not create a
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

## Create and run a photo avatar

Restart the app and expand **Live 3D avatar (prototype)**. Upload one upright,
head-to-toe image of one person with the face clearly visible, arms and legs
unobstructed, and some space between the arms and torso. A higher-resolution
image helps retain face detail. Close the webcam tracker, games, and other
GPU-heavy software first. Choose **512** texture detail initially; higher
detail does not reduce the shape model's full inference memory. Click
**Create & Load My Live Avatar**. The app generates the textured mesh, runs
Blender's automatic rigging step, and loads the prepared GLB into the live
preview. Then press **Start webcam tracking** to grant camera access and begin
tracking.

Record the reconstruction and Blender rigging/validation durations shown in the
completion status. While the webcam is running, record the **Webcam tracking
FPS** metric separately from the page's **Render rate**. Check that the head,
arms, legs, and hands respond, and note any browser-console errors or rigging
warnings before deciding whether the result is usable.

If automatic bone-heat weighting fails, the script tries approximate proximity
weights for the body and hands. This fallback does not animate individual
fingers, and joint bending may need quality improvements. The app checks the
exported skin and verifies that every vertex has weights and that the required
head, arm, hand, and leg joints actually influence vertices. It reports
weighting warnings and which validation failed. The original textured mesh is
kept; when a candidate rigged GLB is available, the error identifies its
location so the failure can be investigated without asking you to run Blender
manually.

This basic fit supports head and limb rotations in the live avatar page; it
does not yet reproduce walking or full-body/foot translation. It does not
create facial expression blendshapes, and photo geometry may not bend cleanly
at joints.

The reported peak is PyTorch memory allocated by the model process; it is not
the GPU's total usage across all applications. The workflow logs and the
Windows Task Manager GPU memory graph can help identify memory used elsewhere.

Even a clear full-body front photo cannot provide accurate side, back, or
occluded details. A small or blurry face in the reference can reduce facial
likeness. This path does not add realistic facial expressions or recover
occluded geometry.
