# FasterLivePortrait on Replit

## Run the web UI

The `Start application` workflow runs:

```bash
python webui.py --mode onnx --host_ip 0.0.0.0 --port 5000
```

Use the Replit Preview pane rather than a localhost URL.

## Runtime setup

- Python dependencies are declared in `requirements.txt`.
- FFmpeg and required native libraries are configured in `.replit`.
- Model checkpoints are downloaded into the git-ignored `checkpoints/` directory:
  - `hf download warmshao/FasterLivePortrait --local-dir checkpoints`
  - `hf download hexgrad/Kokoro-82M --local-dir checkpoints/Kokoro-82M`

Run those download commands again if the workspace checkpoint cache is cleared.

## Inference mode and performance

This workspace has no accessible NVIDIA driver. The workflow therefore uses ONNX Runtime's `CPUExecutionProvider`. The UI is functional, but loading models and generating animations can be slow and is not real-time.

TensorRT remains available as an explicit project mode for a compatible NVIDIA GPU environment:

```bash
python webui.py --mode trt --host_ip 0.0.0.0 --port 5000
```

That mode additionally requires TensorRT 8, converted `.trt` engines, and the matching GridSample3D native plugin. Installing CUDA Python wheels alone does not provide those GPU capabilities.