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

The setup installs PyAV in the isolated MimicMotion environment. If a render
finishes inference but reports that PyAV is missing while writing the MP4, run
this once from the project root, then rerun the benchmark:

```bat
third_party\MimicMotion\.venv\Scripts\python.exe -m pip install av==12.0.0
```

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
output FPS in one report, run from the project root. The default matches the
driver's original timing:

```bat
python benchmark_full_body.py ^
  --source-image "C:\path\to\full-body-avatar.png" ^
  --driving-video "C:\path\to\webcam-pose.mp4" ^
  --output-dir results\full_body_benchmark
```

The report is written to `results\full_body_benchmark\benchmark.json`. It
samples `nvidia-smi` once per second and preserves the backend log tail so an
OOM or DWPose error is visible in the report. The report includes driver and
output duration/FPS metadata plus a `timing` comparison with the duration
difference, ratio, and a boolean indicating whether the output is within two
frames (or 0.1 seconds) of the driver's duration.

The default 8 GB settings are 512 px, 16-frame temporal chunks, 4-frame
overlap, and 15 denoising steps. Output FPS defaults to matching the driver's
sampled frame rate, preserving playback duration instead of stretching clips
to a fixed 15 FPS. In the UI, leave **Output FPS (0 = match driver timing)** at
0 to preserve the driver's speed. For an intentional FPS override, set a value
in the UI or pass `--fps`, for example:

```bat
python full_body.py ^
  --source-image "C:\path\to\full-body-avatar.png" ^
  --driving-video "C:\path\to\webcam-pose.mp4" ^
  --output-dir results\full_body ^
  --fps 15
```

The application launches a
small runner that enables Diffusers sequential CPU model offload before
inference, so the VAE and other components do not all remain resident on the
8 GB GPU.

### Compare detail settings on the Windows GPU

The default denoising-step count remains 15. To compare the existing setting
with 20 steps at the lower-memory 384 px resolution, use the same target image
and driver clip for both runs:

```bat
python benchmark_full_body.py ^
  --source-image "C:\path\to\full-body-avatar.png" ^
  --driving-video "C:\path\to\webcam-pose.mp4" ^
  --output-dir results\full_body_quality_384 ^
  --resolution 384 ^
  --compare-steps 15 20 ^
  --sample-seconds 0.25
```

This runs the settings sequentially in separate `steps_15` and `steps_20`
folders. Each `benchmark.json` records denoising steps, wall time, sampled peak
VRAM, and output video metadata. Each run saves `quality-samples.png` with three
evenly spaced full-frame samples. When the isolated MimicMotion DWPose
environment finds enough confident landmarks, it also saves `quality-details.png`
with enlarged face and hand crops from those same frames. Uncertain or ambiguous
regions are omitted; no crop is an automatic quality score. The parent folder
receives `quality-comparison.json`, which links both reports and both sample
sheets. Add `--visual-notes "..."` to save your observations in the comparison
report.
Review the samples and VRAM headroom on the Alienware before changing the
default; a lower render resolution or extra denoising steps are not assumed to
improve quality without that measured review.

If a 512 px render runs out of GPU memory, retry at 384 px while keeping the
other settings unchanged. The benchmark accepts `--resolution 384`; the
default remains 512 px.

Before generation, the UI's **CHECK FRAMING, PEOPLE & DURATION** action checks
target dimensions/aspect ratio, driver duration, and up to 12 evenly spaced
driver frames with DWPose Wholebody in the isolated MimicMotion environment.
It also keeps the target image visible beside up to four evenly spaced driver
frames with timestamp captions. The target keeps its original aspect ratio;
driver frames are only scaled down to fit and are not cropped. Select a driver
thumbnail to inspect it at full size. Use both panels to check head, hands,
feet, clothing, and frame margins before confirming.
It warns about multiple people, missing people, missing head/arm/leg landmarks,
landscape or unusually cropped-looking target framing, low-resolution targets,
clips under one second, clips over 30 seconds, and unreadable videos. Sparse
pose checking runs on CPU and does not start GPU rendering. If its model assets
are unavailable, the report says so and asks for a manual clip review. Changing
the target or driver clears the confirmation checkbox. The user must still
review the report and confirm before the render can start; the full-body path
never silently falls back to the face-only renderer.

#### Check preview margins on Windows

In the Windows app, repeat the preflight with one portrait and one landscape
target/driver pair. Confirm that all four edges remain visible in both preview
panels, that the target and driver previews remain after the report updates,
and that changing either input clears the confirmation checkbox.

## Important limitations

- This is not live webcam output. Record first, then render.
- A headshot cannot create target-owned hands or clothing; use a full-body
  reference with the relevant body parts visible.
- Pose generation can fail when the driver is occluded, out of frame, or
  contains multiple people.
- ControlNet-style pose conditioning does not guarantee perfect hands. The
  backend is isolated so a newer pose-video model can replace MimicMotion
  without changing the application UI.