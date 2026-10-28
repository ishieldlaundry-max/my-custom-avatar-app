import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from src.pipelines.full_body_animation_pipeline import (
    FullBodyAnimationCancelled,
    FullBodyAnimationError,
    FullBodyRenderOptions,
    MimicMotionBackend,
    estimate_full_body_output_timing,
    extract_driver_preview,
    extract_driver_preview_frames,
    format_dwp_pose_warnings,
    inspect_full_body_inputs,
    match_driver_output_fps,
    _read_driver_fps,
)
from src.utils.gradio_file_links import format_cancelled_render_status


def _fake_backend(tmp_path: Path) -> tuple[MimicMotionBackend, Path, Path]:
    root = tmp_path / "MimicMotion"
    (root / "models" / "DWPose").mkdir(parents=True)
    (root / "inference.py").write_text("# upstream placeholder\n", encoding="utf-8")
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--inference_config")
parser.add_argument("--output_dir")
args = parser.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
(Path(args.output_dir) / "fake-output.mp4").write_bytes(b"mp4")
""",
        encoding="utf-8",
    )
    for relative in (
        "models/DWPose/yolox_l.onnx",
        "models/DWPose/dw-ll_ucoco_384.onnx",
        "models/MimicMotion_1-1.pth",
    ):
        (root / relative).write_bytes(b"model")
    source_image = tmp_path / "target.png"
    driving_video = tmp_path / "driver.mp4"
    source_image.write_bytes(b"image")
    driving_video.write_bytes(b"video")
    return (
        MimicMotionBackend(
            root=root,
            python_executable=Path(sys.executable),
            runner_script=runner,
        ),
        source_image,
        driving_video,
    )


def test_8gb_defaults_are_16_frame_and_validate():
    options = FullBodyRenderOptions()
    options.validate()
    assert options.resolution == 512
    assert options.chunk_size == 16
    assert options.num_inference_steps == 15
    assert options.fps is None


def test_384_resolution_is_supported_for_lower_vram_trials():
    FullBodyRenderOptions(resolution=384).validate()


def test_unsupported_resolution_is_rejected():
    with pytest.raises(ValueError, match="384, 512, or 576"):
        FullBodyRenderOptions(resolution=448).validate()


def test_read_driver_fps_uses_ffprobe_metadata(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *, capture_output, text, check):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"streams":[{"avg_frame_rate":"30000/1001","r_frame_rate":"30/1"}]}',
            stderr="",
        )

    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline.subprocess.run",
        fake_run,
    )

    assert _read_driver_fps(tmp_path / "driver.mp4") == pytest.approx(
        29.97,
        abs=0.001,
    )
    assert commands[0][0] == "ffprobe"
    assert commands[0][commands[0].index("-show_entries") + 1] == (
        "stream=avg_frame_rate,r_frame_rate"
    )


def test_8gb_backend_writes_config_and_returns_generated_video(tmp_path, monkeypatch):
    backend, source_image, driving_video = _fake_backend(tmp_path)
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_fps",
        lambda _path: 30.0,
    )

    output = backend.render(
        source_image,
        driving_video,
        tmp_path / "output",
    )

    assert output.name == "fake-output.mp4"
    config = (tmp_path / "output" / "mimicmotion_full_body.yaml").read_text(
        encoding="utf-8"
    )
    assert "num_frames: 16" in config
    assert "resolution: 512" in config
    assert "frames_overlap: 4" in config
    assert "sample_stride: 1" in config
    assert "fps: 30.0" in config


def test_60fps_driver_keeps_base_stride_and_uses_30fps_output(tmp_path, monkeypatch):
    backend, source_image, driving_video = _fake_backend(tmp_path)
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_fps",
        lambda _path: 60.0,
    )

    backend.render(
        source_image,
        driving_video,
        tmp_path / "output",
    )

    config = (tmp_path / "output" / "mimicmotion_full_body.yaml").read_text(
        encoding="utf-8"
    )
    assert "sample_stride: 1" in config
    assert "fps: 30.0" in config


@pytest.mark.parametrize(
    ("driver_fps", "expected_output_fps"),
    [(30, 30), (29.97, 29.97), (48, 24), (60, 30)],
)
def test_default_output_fps_preserves_driver_timing(driver_fps, expected_output_fps):
    assert match_driver_output_fps(driver_fps) == expected_output_fps


def test_timing_estimate_matches_driver_duration_by_default(tmp_path, monkeypatch):
    driver = tmp_path / "driver.mp4"
    driver.write_bytes(b"video")
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_fps",
        lambda _path: 60.0,
    )
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_duration",
        lambda _path: 8.0,
    )

    estimate = estimate_full_body_output_timing(driver, 0)

    assert "Detected driver FPS:** 60 FPS" in estimate
    assert "Output FPS:** 30 FPS (matched to driver timing)" in estimate
    assert "Estimated output duration:** 8.0 seconds" in estimate


def test_timing_estimate_updates_duration_for_fps_override(tmp_path, monkeypatch):
    driver = tmp_path / "driver.mp4"
    driver.write_bytes(b"video")
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_fps",
        lambda _path: 30.0,
    )
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_duration",
        lambda _path: 8.0,
    )

    estimate = estimate_full_body_output_timing(driver, 15)

    assert "Output FPS:** 15 FPS (override)" in estimate
    assert "Estimated output duration:** 16.0 seconds" in estimate


def test_timing_estimate_explains_unavailable_metadata(tmp_path, monkeypatch):
    driver = tmp_path / "driver.mp4"
    driver.write_bytes(b"video")
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_fps",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline._read_driver_duration",
        lambda _path: 8.0,
    )

    estimate = estimate_full_body_output_timing(driver, 24)

    assert "Detected driver FPS:** unavailable" in estimate
    assert "Output FPS:** 24 FPS (override)" in estimate
    assert (
        "Estimated output duration:** unavailable because driver FPS metadata "
        "could not be read" in estimate
    )


def test_timing_estimate_requests_driver_clip_when_missing():
    assert "Upload or record a driver clip" in estimate_full_body_output_timing(None, 0)


def test_missing_pyav_reports_recovery_command(tmp_path):
    backend, source_image, driving_video = _fake_backend(tmp_path)
    backend.runner_script.write_text(
        "print('ImportError: PyAV is not installed, and is necessary for the video operations in torchvision.')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    with pytest.raises(FullBodyAnimationError, match="completed inference") as error:
        backend.render(
            source_image,
            driving_video,
            tmp_path / "output",
            FullBodyRenderOptions(fps=15),
        )

    assert "pip install av==12.0.0" in str(error.value)


def test_cancel_stops_runner_and_preserves_log_and_existing_output(tmp_path):
    backend, source_image, driving_video = _fake_backend(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    completed_output = output_dir / "completed-before-cancel.mp4"
    completed_output.write_bytes(b"completed")
    started_marker = output_dir / "runner-started"
    run_count_path = output_dir / "runner-count"
    backend.runner_script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output_dir') + 1])\n"
        "count_path = out / 'runner-count'\n"
        "count = int(count_path.read_text()) + 1 if count_path.exists() else 1\n"
        "count_path.write_text(str(count))\n"
        "print(f'runner attempt {count} started', flush=True)\n"
        "if count == 1:\n"
        "    (out / 'runner-started').write_text('yes')\n"
        "    time.sleep(30)\n"
        "else:\n"
        "    (out / 'retry-output.mp4').write_bytes(b'retry output')\n",
        encoding="utf-8",
    )
    result = {}

    def run_render():
        try:
            backend.render(
                source_image,
                driving_video,
                output_dir,
                FullBodyRenderOptions(fps=15),
            )
        except Exception as exc:  # captured for assertions in the test thread
            result["error"] = exc

    render_thread = threading.Thread(target=run_render)
    render_thread.start()
    deadline = time.monotonic() + 5
    cancelled = False
    started = False
    while time.monotonic() < deadline:
        if started_marker.exists():
            started = True
            with pytest.raises(FullBodyAnimationError, match="already running"):
                backend.render(
                    source_image,
                    driving_video,
                    output_dir,
                    FullBodyRenderOptions(fps=15),
                )
            cancelled = backend.cancel_active_render()
            break
        time.sleep(0.05)
    render_thread.join(timeout=5)

    assert started
    assert cancelled
    assert not render_thread.is_alive()
    cancelled_error = result.get("error")
    assert isinstance(cancelled_error, FullBodyAnimationCancelled)
    assert completed_output.read_bytes() == b"completed"
    attempt_logs = sorted(output_dir.glob("mimicmotion-*.log"))
    assert len(attempt_logs) == 1
    cancelled_log = attempt_logs[0]
    assert "runner attempt 1 started" in cancelled_log.read_text(encoding="utf-8")
    assert str(cancelled_log) in str(cancelled_error)
    assert cancelled_error.log_path == cancelled_log.resolve()

    retry_output = backend.render(
        source_image,
        driving_video,
        output_dir,
        FullBodyRenderOptions(fps=15),
    )
    assert retry_output.name == "retry-output.mp4"

    attempt_logs = sorted(output_dir.glob("mimicmotion-*.log"))
    assert len(attempt_logs) == 2
    assert cancelled_log.read_text(encoding="utf-8").find("runner attempt 1 started") >= 0
    retry_log = next(path for path in attempt_logs if path != cancelled_log)
    assert "runner attempt 2 started" in retry_log.read_text(encoding="utf-8")
    assert str(cancelled_log) in str(cancelled_error)


def test_cancelled_status_links_only_the_retained_log_with_spaces(tmp_path):
    output_dir = tmp_path / "Jane Doe" / "full body renders"
    output_dir.mkdir(parents=True)
    log_path = output_dir / "mimicmotion-attempt-1.log"
    log_path.write_text("cancelled runner output", encoding="utf-8")
    allowed_paths = ["assets"]
    message = (
        f"Full-body render cancelled. This attempt's log is preserved at {log_path}. "
        f"Existing and partial output files remain in {output_dir}."
    )

    status = format_cancelled_render_status(
        message,
        log_path,
        allowed_paths=allowed_paths,
        allowed_root=output_dir,
    )

    encoded_path = str(log_path.resolve()).replace(" ", "%20")
    assert "**Render cancelled.**" in status
    assert f"[`{log_path.resolve()}`](/gradio_api/file={encoded_path})" in status
    assert "Existing and partial output files remain" in status
    assert "Jane Doe" in status and "full body renders" in status
    assert allowed_paths == ["assets", str(log_path.resolve())]


def test_cancelled_status_does_not_link_log_outside_allowed_root(tmp_path):
    allowed_root = tmp_path / "results" / "full_body"
    outside_log = tmp_path / "private" / "mimicmotion-](javascript.log"
    outside_log.parent.mkdir(parents=True)
    outside_log.write_text("private", encoding="utf-8")
    allowed_paths = []
    message = f"Render cancelled. Log is at {outside_log}."

    status = format_cancelled_render_status(
        message,
        outside_log,
        allowed_paths=allowed_paths,
        allowed_root=allowed_root,
    )

    assert f"[`{outside_log.resolve()}`]" not in status
    assert r"javascript\.log" in status
    assert "](" not in status
    assert allowed_paths == []


def test_missing_model_assets_are_actionable(tmp_path):
    root = tmp_path / "MimicMotion"
    root.mkdir()
    source = tmp_path / "target.png"
    driver = tmp_path / "driver.mp4"
    source.write_bytes(b"image")
    driver.write_bytes(b"video")

    with pytest.raises(FullBodyAnimationError, match="MimicMotion is not installed"):
        MimicMotionBackend(root=root).preflight(source, driver)


def test_input_report_warns_about_landscape_target_and_unreadable_driver(tmp_path):
    source = tmp_path / "target.png"
    Image.new("RGB", (1200, 800), "white").save(source)
    driver = tmp_path / "driver.mp4"
    driver.write_bytes(b"not a video")

    report = inspect_full_body_inputs(source, driver)

    assert report.target_width == 1200
    assert report.target_height == 800
    assert report.has_warnings
    assert any("landscape" in warning for warning in report.warnings)
    assert any("duration could not be read" in warning for warning in report.warnings)
    assert "CHECK" not in report.to_markdown()


def test_input_report_accepts_missing_files_without_starting_generation(tmp_path):
    report = inspect_full_body_inputs(
        tmp_path / "missing-target.png",
        tmp_path / "missing-driver.mp4",
    )

    assert report.has_warnings
    assert "not available for inspection" in report.to_markdown()


def test_pose_report_warns_about_multiple_people_and_missing_landmarks():
    warnings = format_dwp_pose_warnings(
        {
            "sample_count": 2,
            "frames": [
                {
                    "people": [
                        {"missing_landmarks": []},
                        {"missing_landmarks": ["left ankle", "right wrist"]},
                    ]
                },
                {"people": [{"missing_landmarks": ["left ankle"]}]},
            ],
        }
    )

    assert any("multiple people in 1 of 2" in warning for warning in warnings)
    assert any("left ankle" in warning and "right wrist" in warning for warning in warnings)


def test_pose_report_warns_when_no_person_is_detected():
    warnings = format_dwp_pose_warnings(
        {"sample_count": 2, "frames": [{"people": []}, {"people": []}]}
    )

    assert len(warnings) == 1
    assert "detected no person" in warnings[0]


def test_pose_report_warns_when_person_disappears_from_some_samples():
    warnings = format_dwp_pose_warnings(
        {
            "sample_count": 2,
            "frames": [
                {"people": [{"missing_landmarks": []}]},
                {"people": []},
            ],
        }
    )

    assert any("no person in 1 of 2" in warning for warning in warnings)


def test_pose_report_accepts_one_complete_person():
    warnings = format_dwp_pose_warnings(
        {"sample_count": 3, "frames": [{"people": [{"missing_landmarks": []}]}] * 3}
    )

    assert len(warnings) == 1
    assert "found one person" in warnings[0]


def test_short_clip_preflight_catches_a_brief_person_between_legacy_samples(
    monkeypatch, tmp_path
):
    import contextlib
    import types

    import numpy as np

    from scripts.mimicmotion_pose_preflight import (
        inspect_video,
        select_sample_indices,
    )

    frame_count = 60
    legacy_indices = np.linspace(0, frame_count - 1, num=12, dtype=np.int64)
    sample_indices = select_sample_indices(frame_count, 30.0, 12)
    twelve_frame_indices = select_sample_indices(frame_count, 30.0, 6)
    assert 17 not in legacy_indices
    assert 17 in sample_indices
    assert len(sample_indices) == 24
    assert len(twelve_frame_indices) == 12
    assert len(select_sample_indices(1800, 30.0, 12)) == 12

    class FakeReader:
        def __len__(self):
            return frame_count

        def get_avg_fps(self):
            return 30.0

        def __getitem__(self, frame_index):
            frame = np.full((2, 2, 3), frame_index, dtype=np.uint8)
            return types.SimpleNamespace(asnumpy=lambda: frame)

    class FakeDetector:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, frame):
            person_count = 2 if int(frame[0, 0, 0]) == 17 else 1
            subset = [np.zeros(14, dtype=np.float32) for _ in range(person_count)]
            return {"bodies": {"subset": subset}}

        def release_memory(self):
            pass

    decord = types.ModuleType("decord")
    decord.cpu = lambda _index: "cpu"
    decord.VideoReader = lambda _path, ctx: FakeReader()
    detector_module = types.ModuleType("mimicmotion.dwpose.dwpose_detector")
    detector_module.DWposeDetector = FakeDetector
    monkeypatch.setitem(sys.modules, "decord", decord)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, "mimicmotion", types.ModuleType("mimicmotion"))
    monkeypatch.setitem(sys.modules, "mimicmotion.dwpose", types.ModuleType("mimicmotion.dwpose"))
    monkeypatch.setitem(sys.modules, "mimicmotion.dwpose.dwpose_detector", detector_module)

    report = inspect_video(tmp_path / "brief-person.mp4", tmp_path, max_samples=12)

    brief_appearance = next(
        frame for frame in report["frames"] if frame["frame_index"] == 17
    )
    assert len(brief_appearance["people"]) == 2
    assert report["sample_count"] == 24
    assert report["elapsed_seconds"] >= 0


def test_pose_check_runs_in_isolated_environment_and_reports_detection(tmp_path):
    root = tmp_path / "MimicMotion"
    model_dir = root / "models" / "DWPose"
    model_dir.mkdir(parents=True)
    (model_dir / "yolox_l.onnx").write_bytes(b"detector")
    (model_dir / "dw-ll_ucoco_384.onnx").write_bytes(b"pose")
    python = Path(sys.executable)
    runner = tmp_path / "pose_preflight.py"
    runner.write_text(
        "import json\n"
        "print(json.dumps({'sample_count': 1, 'frames': [{'people': ["
        "{'missing_landmarks': []}, {'missing_landmarks': []}]}]}))\n",
        encoding="utf-8",
    )
    video = tmp_path / "driver.mp4"
    video.write_bytes(b"video")
    backend = MimicMotionBackend(
        root=root,
        python_executable=python,
        pose_preflight_runner=runner,
    )

    warnings = backend.inspect_driver_video(video)

    assert "multiple people" in warnings[0]
    assert any(
        "DWPose preflight elapsed:" in warning and "1 sampled frames" in warning
        for warning in warnings
    )


def test_pose_check_reports_the_configured_timeout(tmp_path, monkeypatch):
    root = tmp_path / "MimicMotion"
    model_dir = root / "models" / "DWPose"
    model_dir.mkdir(parents=True)
    (model_dir / "yolox_l.onnx").write_bytes(b"detector")
    (model_dir / "dw-ll_ucoco_384.onnx").write_bytes(b"pose")
    python = Path(sys.executable)
    runner = tmp_path / "slow_preflight.py"
    runner.write_text(
        "import sys, time\n"
        "print('DWPose progress: 2/12', file=sys.stderr, flush=True)\n"
        "time.sleep(2)\n",
        encoding="utf-8",
    )
    video = tmp_path / "driver.mp4"
    video.write_bytes(b"video")
    backend = MimicMotionBackend(
        root=root,
        python_executable=python,
        pose_preflight_runner=runner,
    )
    monkeypatch.setattr(
        "src.pipelines.full_body_animation_pipeline.DWPOSE_PREFLIGHT_TIMEOUT_SECONDS",
        0.5,
    )

    warnings = backend.inspect_driver_video(video)

    assert any("timed out after 0.5 seconds" in warning for warning in warnings)
    assert any("Last reported status: DWPose progress: 2/12." in warning for warning in warnings)


def test_pose_check_warns_actionably_when_dwpose_assets_are_missing(tmp_path):
    root = tmp_path / "MimicMotion"
    root.mkdir()
    video = tmp_path / "driver.mp4"
    video.write_bytes(b"video")
    backend = MimicMotionBackend(root=root, python_executable=Path(sys.executable))

    warnings = backend.inspect_driver_video(video)

    assert len(warnings) == 1
    assert "Run setup_full_body_windows.bat" in warnings[0]


def test_pose_check_handles_missing_video_without_raising(tmp_path):
    backend = MimicMotionBackend(root=tmp_path)

    warnings = backend.inspect_driver_video(None)

    assert "upload a driver video" in warnings[0]


def test_extract_driver_preview_decodes_middle_frame_and_preserves_full_frame(
    monkeypatch, tmp_path
):
    import cv2
    import numpy as np
    import src.pipelines.full_body_animation_pipeline as pipeline

    video = tmp_path / "driver.mp4"
    video.write_bytes(b"placeholder")
    source_frame = np.zeros((200, 100, 3), dtype=np.uint8)
    source_frame[:, :, 2] = 255

    class FakeCapture:
        def __init__(self, _path):
            self.position = None
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 9 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.position = value
            return True

        def read(self):
            return True, source_frame.copy()

        def release(self):
            self.released = True

    capture = FakeCapture(str(video))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    preview = extract_driver_preview(video)

    assert capture.position == 4
    assert capture.released
    assert preview.shape == (200, 100, 3)
    assert tuple(preview[0, 0]) == (255, 0, 0)


def test_extract_driver_preview_downscales_large_frames_without_cropping(
    monkeypatch, tmp_path
):
    import cv2
    import numpy as np
    import src.pipelines.full_body_animation_pipeline as pipeline

    video = tmp_path / "large-driver.mp4"
    video.write_bytes(b"placeholder")
    source_frame = np.zeros((1200, 1920, 3), dtype=np.uint8)

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, _prop):
            return 1

        def set(self, _prop, _value):
            return True

        def read(self):
            return True, source_frame.copy()

        def release(self):
            pass

    monkeypatch.setattr(
        cv2, "VideoCapture", lambda _path: FakeCapture()
    )

    preview = extract_driver_preview(video, max_edge=960)

    assert preview.shape == (600, 960, 3)


def test_extract_driver_preview_frames_samples_evenly_spaced_moments(
    monkeypatch, tmp_path
):
    import cv2
    import numpy as np

    video = tmp_path / "moments.mp4"
    video.write_bytes(b"placeholder")
    capture_instances = []

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0
            self.positions_read = []
            self.released = False
            capture_instances.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10
            if prop == cv2.CAP_PROP_FPS:
                return 10
            return 0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)
            return True

        def read(self):
            self.positions_read.append(self.position)
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            frame[:, :, 0] = self.position
            return True, frame

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    previews = extract_driver_preview_frames(video)

    assert len(previews) == 4
    assert capture_instances[0].positions_read == [0, 3, 6, 9]
    assert capture_instances[0].released
    assert [label for _, label in previews] == ["0.0 s", "0.3 s", "0.6 s", "0.9 s"]
    assert [tuple(frame[0, 0]) for frame, _ in previews] == [
        (0, 0, 0),
        (0, 0, 3),
        (0, 0, 6),
        (0, 0, 9),
    ]
    assert all(frame.shape == (40, 60, 3) for frame, _ in previews)


@pytest.mark.parametrize(
    ("height", "width"),
    [(1200, 800), (800, 1200)],
    ids=["portrait", "landscape"],
)
def test_extract_driver_preview_frames_preserves_all_margins(
    monkeypatch, tmp_path, height, width
):
    import cv2
    import numpy as np

    video = tmp_path / "framed.mp4"
    video.write_bytes(b"placeholder")
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    markers = [
        ((10, 20, 30), (30, 20, 10)),
        ((40, 50, 60), (60, 50, 40)),
        ((70, 80, 90), (90, 80, 70)),
        ((100, 110, 120), (120, 110, 100)),
    ]
    frame[:40, :40] = markers[0][0]
    frame[:40, -40:] = markers[1][0]
    frame[-40:, :40] = markers[2][0]
    frame[-40:, -40:] = markers[3][0]

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1
            if prop == cv2.CAP_PROP_FPS:
                return 30
            return 0

        def set(self, _prop, _value):
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture())
    previews = extract_driver_preview_frames(video, max_frames=1, max_edge=600)

    preview = previews[0][0]
    scale = min(1.0, 600 / max(height, width))
    assert preview.shape == (
        round(height * scale),
        round(width * scale),
        3,
    )
    for point, (_, expected_rgb) in zip(
        [(4, 4), (4, -5), (-5, 4), (-5, -5)],
        markers,
        strict=True,
    ):
        assert tuple(preview[point[0], point[1]]) == expected_rgb


def test_extract_driver_preview_handles_missing_or_unreadable_video(tmp_path):
    assert extract_driver_preview(tmp_path / "missing.mp4") is None
    assert extract_driver_preview_frames(tmp_path / "missing.mp4") == []