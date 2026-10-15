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
    inspect_full_body_inputs,
)


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


def test_384_resolution_is_supported_for_lower_vram_trials():
    FullBodyRenderOptions(resolution=384).validate()


def test_unsupported_resolution_is_rejected():
    with pytest.raises(ValueError, match="384, 512, or 576"):
        FullBodyRenderOptions(resolution=448).validate()


def test_8gb_backend_writes_config_and_returns_generated_video(tmp_path):
    backend, source_image, driving_video = _fake_backend(tmp_path)

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


def test_missing_pyav_reports_recovery_command(tmp_path):
    backend, source_image, driving_video = _fake_backend(tmp_path)
    backend.runner_script.write_text(
        "print('ImportError: PyAV is not installed, and is necessary for the video operations in torchvision.')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    with pytest.raises(FullBodyAnimationError, match="completed inference") as error:
        backend.render(source_image, driving_video, tmp_path / "output")

    assert "pip install av==12.0.0" in str(error.value)


def test_cancel_stops_runner_and_preserves_log_and_existing_output(tmp_path):
    backend, source_image, driving_video = _fake_backend(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    completed_output = output_dir / "completed-before-cancel.mp4"
    completed_output.write_bytes(b"completed")
    started_marker = output_dir / "runner-started"
    backend.runner_script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output_dir') + 1])\n"
        "print('runner started', flush=True)\n"
        "(out / 'runner-started').write_text('yes')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    result = {}

    def run_render():
        try:
            backend.render(source_image, driving_video, output_dir)
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
                backend.render(source_image, driving_video, output_dir)
            assert "runner started" in (output_dir / "mimicmotion.log").read_text(
                encoding="utf-8"
            )
            cancelled = backend.cancel_active_render()
            break
        time.sleep(0.05)
    render_thread.join(timeout=5)

    assert started
    assert cancelled
    assert not render_thread.is_alive()
    assert isinstance(result.get("error"), FullBodyAnimationCancelled)
    assert completed_output.read_bytes() == b"completed"
    assert "runner started" in (output_dir / "mimicmotion.log").read_text(
        encoding="utf-8"
    )


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