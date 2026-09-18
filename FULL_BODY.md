# Offline full-body avatar animation

The normal FasterLivePortrait path is face-only. The optional full-body path
uses [Tencent MimicMotion](https://github.com/Tencent/MimicMotion), which
includes DWPose Wholebody preprocessing for body, face, and hand keypoints and
a temporal reference-conditioned video generator. The reference image remains
the appearance source; the webcam recording supplies pose.

## Requirements

- Windows Alienware with an NVIDIA GPU
- A full-body target image with the target clothing and hands visible
- A short recorded webcam video with one person and a clear full-body pose
- Approximately 8 GB VRAM minimum for the 16-frame preset
- Disk space for the MimicMotion checkpoint and Stable Video Diffusion base model
- Hugging Face access to `stabilityai/stable-video-diffusion-img2vid-xt-1-1`

The 8 GB preset is deliberately offline and slow. The normal face mode and
virtual-camera mode are unchanged.

## Install on Windows

From the project root, run:

```bat
setup_full_body_windows.bat
```

This creates `third_party\MimicMotion\.venv`, downloads DWPose and the
MimicMotion checkpoint, and leaves the existing FasterLivePortrait environment
untouched. The Stable Video Diffusion base model downloads from Hugging Face on
the first render. It is gated, so accept its access terms while signed in at
the model page, then authenticate the isolated environment locally:

```bat
third_party\MimicMotion\.venv\Scripts\huggingface-cli.exe login
```

Enter the Hugging Face token only into that local prompt; do not paste it into
chat or commit it to the project.

## Render a full-body avatar

Record a short webcam clip first, then run:

```bat
.\third_party\MimicMotion\.venv\Scripts\python.exe full_body.py ^
  --source-image "C:\path\to\full-body-avatar.png" ^
  --driving-video "C:\path\to\webcam-pose.mp4" ^
  --output-dir results\full_body
```

To capture peak VRAM, wall-clock render time, output duration, frame count, and
output FPS in one report, run from the project root:

```bat
python benchmark_full_body.py ^
  --source-image "C:\path\to\full-body-avatar.png" ^
  --driving-video "C:\path\to\webcam-pose.mp4" ^
  --output-dir results\full_body_benchmark
```

The report is written to `results\full_body_benchmark\benchmark.json`. It
samples `nvidia-smi` once per second and preserves the backend log tail so an
OOM or DWPose error is visible in the report.

The default 8 GB settings are 512 px, 16-frame temporal chunks, 4-frame
overlap, 15 denoising steps, and 15 FPS output. The application launches a
small runner that enables Diffusers sequential CPU model offload before
inference, so the VAE and other components do not all remain resident on the
8 GB GPU.

Before generation, the UI's **CHECK FRAMING & DURATION** action checks the
target dimensions/aspect ratio and reads the driver duration. It warns about
landscape or unusually cropped-looking framing, low-resolution targets, clips
under one second, clips over 30 seconds, and unreadable videos. These checks
cannot prove that a person is fully visible, so confirm that the target shows
the head, feet, and both hands before enabling the render confirmation. The
full-body path never silently falls back to the face-only renderer.

## Important limitations

- This is not live webcam output. Record first, then render.
- A headshot cannot create target-owned hands or clothing; use a full-body
  reference with the relevant body parts visible.
- Pose generation can fail when the driver is occluded, out of frame, or
  contains multiple people.
- ControlNet-style pose conditioning does not guarantee perfect hands. The
  backend is isolated so a newer pose-video model can replace MimicMotion
  without changing the application UI.