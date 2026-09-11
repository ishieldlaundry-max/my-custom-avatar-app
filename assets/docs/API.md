## FasterLivePortrait API Usage Guide

### Building the Image
* Decide on an image name, for example `shaoguo/faster_liveportrait_api:v1.0`. Replace the `-t` parameter in the following command with your chosen name.
* Run `docker build -t shaoguo/faster_liveportrait_api:v1.0 -f DockerfileAPI .`

### Running the Image
Ensure that your machine has Nvidia GPU drivers installed. CUDA version should be 12.0 or higher. Two scenarios are described below.

* Running on a Local Machine (typically for self-testing)
  * Modify the image name according to what you defined above.
  * Confirm the service port number, default is `9871`. You can define your own by changing the `SERVER_PORT` environment variable in the command below. Remember to also change `-p 9871:9871` to map the port.
  * Set the model path environment variable `CHECKPOINT_DIR`. If you've previously downloaded FasterLivePortrait's onnx model and converted it to trt, I recommend mapping the model files into the container using `-v`, for example `-v E:\my_projects\FasterLivePortrait\checkpoints:/root/FasterLivePortrait/checkpoints`. This avoids re-downloading the onnx model and doing trt conversion. Otherwise, I will check if `CHECKPOINT_DIR` has models, and if not, I will automatically download (ensure network connectivity) and do trt conversion, which will take considerable time.
  * Run command (note: modify the following command according to your settings):
    ```shell
    docker run -d --gpus=all \
    --name faster_liveportrait_api \
    -v E:\my_projects\FasterLivePortrait\checkpoints:/root/FasterLivePortrait/checkpoints \
    -e CHECKPOINT_DIR=/root/FasterLivePortrait/checkpoints \
    -e SERVER_PORT=9871 \
    -p 9871:9871 \
    --restart=always \
    shaoguo/faster_liveportrait_api:v1.0 \
    /bin/bash
    ```
  * Normal operation should display the following information(docker logs $container_id). The running logs are saved in `/root/FasterLivePortrait/logs/log_run.log`:
    ```shell
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:9871 (Press CTRL+C to quit)
    ```

* Running on Cloud GPU Cluster (production environment)
  * This needs to be configured according to different clusters, but the core is the configuration of docker image and environment variables.
  * Load balancing may need to be set up.

### API Call Testing
Refer to `tests/test_api.py`. The default is the Animal model, but now it also supports the Human model.
The return is a compressed package, by default unzipped to `./results/api_*`. Confirm according to the actual printed log.
* `test_with_video_animal()`, image and video driving. Set `flag_pickle=False`. It will additionally return the driving video's pkl file, which can be called directly next time.
* `test_with_pkl_animal()`, image and pkl driving. Because generated driving
  pickles are excluded from version control, the test creates a small neutral
  one-frame `d8.pkl` fixture when it is missing. It requires no model files.
  To regenerate it manually:
  ```shell
  python scripts/generate_pickle_fixture.py
  ```
* `test_with_video_human()`, image and video driving under the Human model, set `flag_is_animal=False`

### Replit and CPU-only verification prerequisites
The API entrypoint currently starts with `configs/trt_infer.yaml` and initializes the TensorRT pipeline during application startup. A successful inference check therefore requires all of the following:

* Python dependencies from `requirements.txt`, plus a compatible PyTorch, CUDA, TensorRT, and TensorRT grid-sample plugin installation.
* An NVIDIA GPU with working drivers and CUDA libraries.
* The FasterLivePortrait checkpoint tree downloaded under `./checkpoints` (or the directory selected with `FLIP_CHECKPOINT_DIR`), with the ONNX files converted to the TensorRT files referenced by `configs/trt_infer.yaml`.
* `ffmpeg` available on `PATH`.

On a CPU-only Replit runtime, `bash scripts/start_api.sh` cannot provide a real inference server because the default mode is TensorRT and `pycuda`/TensorRT require CUDA headers and libraries. Install the complete GPU/model prerequisites before treating a connection refusal as an API result.

Run the checks from the repository root:

```shell
bash scripts/start_api.sh
# in a second shell, after "Application startup complete":
pytest -q tests/test_api.py
```

The pickle test regenerates `assets/examples/driving/d8.pkl` when it is absent.
The video tests use the three-frame
`assets/examples/driving/d0-smoke.mp4` fixture by default so CPU smoke checks
stay within the validation window. The full-length `d0.mp4` path remains
available for production validation:

```shell
FLIP_API_VIDEO_FIXTURE=assets/examples/driving/d0.mp4 pytest -q tests/test_api.py
```

If the smoke fixture is absent, regenerate it with:

```shell
bash scripts/generate_api_smoke_video.sh
```

All integration tests require the API to be running on `127.0.0.1:9871` and
verify a 200 response containing at least one non-empty output video.

### Readiness

Before submitting to `/predict/`, clients can query `GET /health/ready`.
It returns `200` only when inference is ready and `503` while startup is
`starting` or has `failed`. The JSON response includes:

* `status`: `starting`, `ready`, or `failed`
* `backend`: the selected `onnx` or `tensorrt` backend and provider
* `checkpoints.status`: checkpoint initialization progress
* `plugin.status`: TensorRT plugin status, or `not_required` for ONNX
* `error`: a non-secret startup error summary when `status` is `failed`