"""Offline full-body avatar animation through an isolated MimicMotion runtime.

The regular FasterLivePortrait pipeline is intentionally face-only.  This
adapter keeps the body animation dependency isolated because MimicMotion uses
its own PyTorch/diffusers environment and model checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Optional


class FullBodyAnimationError(RuntimeError):
    """A user-actionable full-body rendering error."""


class FullBodyAnimationCancelled(FullBodyAnimationError):
    """Raised after the active MimicMotion child process has been stopped."""

    def __init__(self, message: str, *, log_path: Path | str | None = None):
        super().__init__(message)
        self.log_path = Path(log_path).resolve() if log_path is not None else None


DWPOSE_PREFLIGHT_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class FullBodyRenderOptions:
    """Conservative MimicMotion settings for an 8 GB RTX 4070."""

    resolution: int = 512
    chunk_size: int = 16
    frames_overlap: int = 4
    num_inference_steps: int = 15
    noise_aug_strength: float = 0.0
    guidance_scale: float = 2.0
    sample_stride: int = 1
    fps: Optional[float] = None
    seed: int = 42

    def validate(self) -> None:
        if self.resolution not in (384, 512, 576):
            raise ValueError("Full-body resolution must be 384, 512, or 576.")
        if self.chunk_size != 16:
            raise ValueError("The 8 GB preset requires a 16-frame chunk size.")
        if not 0 <= self.frames_overlap < self.chunk_size:
            raise ValueError("Frame overlap must be between 0 and chunk_size - 1.")
        if not 1 <= self.num_inference_steps <= 50:
            raise ValueError("Inference steps must be between 1 and 50.")
        if self.noise_aug_strength < 0 or self.noise_aug_strength > 1:
            raise ValueError("Noise augmentation must be between 0 and 1.")
        if self.guidance_scale <= 0:
            raise ValueError("Guidance scale must be positive.")
        if self.sample_stride < 1:
            raise ValueError("Sample stride must be at least 1.")
        if self.fps is not None and (
            not math.isfinite(self.fps) or not 1 <= self.fps <= 60
        ):
            raise ValueError("Output FPS must be between 1 and 60, or match the driver.")


@dataclass(frozen=True)
class FullBodyInputReport:
    """Non-destructive checks shown before an expensive full-body render."""

    target_width: Optional[int]
    target_height: Optional[int]
    driver_duration_seconds: Optional[float]
    warnings: tuple[str, ...] = ()

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def to_markdown(self) -> str:
        lines = ["### Full-body input check"]
        if self.target_width and self.target_height:
            ratio = self.target_width / self.target_height
            lines.append(
                f"- Target image: `{self.target_width} × {self.target_height}` "
                f"(aspect ratio `{ratio:.2f}`)"
            )
        else:
            lines.append("- Target image: dimensions could not be read")

        if self.driver_duration_seconds is not None:
            lines.append(
                f"- Driver video: `{self.driver_duration_seconds:.1f} seconds`"
            )
        else:
            lines.append("- Driver video: duration could not be read")

        if self.warnings:
            lines.append("")
            lines.append("**Review before rendering:**")
            lines.extend(f"- {warning}" for warning in self.warnings)
        else:
            lines.append("")
            lines.append(
                "No automated framing or duration warnings. Confirm that the "
                "target shows the head, full body, and both hands."
            )
        return "\n".join(lines)


def format_dwp_pose_warnings(payload: dict) -> tuple[str, ...]:
    """Turn sampled DWPose detections into clear pre-render warnings."""
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return (
            "The automatic pose check returned no sampled frames. "
            "Use a short, clearly visible single-person clip and review it before rendering.",
        )

    sample_count = len(frames)
    people_by_frame = [
        frame.get("people", []) if isinstance(frame, dict) else []
        for frame in frames
    ]
    multiple_frames = sum(len(people) > 1 for people in people_by_frame)
    empty_frames = sum(not people for people in people_by_frame)
    detected_people = [person for people in people_by_frame for person in people]
    if not detected_people:
        return (
            f"DWPose detected no person in {sample_count} sampled frames. "
            "Record a well-lit clip with one person fully visible, including head, "
            "hands, and feet.",
        )

    missing_by_frame = []
    missing_names: set[str] = set()
    for people in people_by_frame:
        frame_missing = set()
        for person in people:
            missing = person.get("missing_landmarks", [])
            if missing:
                frame_missing.update(str(name) for name in missing)
        if frame_missing:
            missing_by_frame.append(frame_missing)
            missing_names.update(frame_missing)

    warnings = []
    if empty_frames:
        warnings.append(
            f"DWPose detected no person in {empty_frames} of {sample_count} "
            "sampled frames. Keep the driver fully in frame throughout the clip."
        )
    if multiple_frames:
        warnings.append(
            f"DWPose found multiple people in {multiple_frames} of {sample_count} "
            "sampled frames. Record with only the intended driver in view to avoid "
            "the pose switching between people."
        )
    if missing_by_frame:
        missing_summary = ", ".join(sorted(missing_names))
        warnings.append(
            f"DWPose missed full-body landmarks in {len(missing_by_frame)} of "
            f"{sample_count} sampled frames ({missing_summary}). Keep the head, "
            "wrists, hips, knees, and ankles visible and unobstructed."
        )
    if not warnings:
        warnings.append(
            f"DWPose found one person with the required full-body landmarks in all "
            f"{sample_count} sampled frames. Still review the clip visually."
        )
    return tuple(warnings)


def _read_target_dimensions(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except (OSError, ValueError):
        return None, None


def _read_driver_duration(path: Path) -> Optional[float]:
    try:
        import ffmpeg

        probe = ffmpeg.probe(str(path))
        video_streams = [
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        duration = (
            video_streams[0].get("duration")
            if video_streams
            else probe.get("format", {}).get("duration")
        )
        return float(duration) if duration is not None else None
    except (ImportError, OSError, ValueError, TypeError, KeyError):
        return None
    except Exception:
        # ffmpeg-python surfaces probe failures as several exception types
        # depending on whether ffprobe or the input codec failed.
        return None


def _read_driver_fps(path: Path) -> Optional[float]:
    """Read source FPS with the ffprobe executable without decoding the clip."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        video_streams = json.loads(result.stdout).get("streams", [])
        if not video_streams:
            return None
        stream = video_streams[0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw_rate = stream.get(key)
            if raw_rate and raw_rate not in ("0/0", "N/A"):
                rate = float(Fraction(raw_rate))
                if math.isfinite(rate) and rate > 0:
                    return rate
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return None
    return None


def match_driver_output_fps(driver_fps: float, sample_stride: int = 1) -> float:
    """Match MimicMotion's sampled-frame timing to the driver's duration."""
    if not math.isfinite(driver_fps) or driver_fps <= 0:
        raise ValueError("Driver FPS must be a positive, finite number.")
    if sample_stride < 1:
        raise ValueError("Sample stride must be at least 1.")
    effective_stride = sample_stride * max(1, int(driver_fps / 24))
    return round(driver_fps / effective_stride, 3)


def estimate_full_body_output_timing(
    driving_video: Optional[str | Path],
    output_fps_override: Optional[float] = None,
) -> str:
    """Describe the detected driver rate and estimated rendered clip duration."""
    try:
        requested_fps = float(output_fps_override or 0)
    except (TypeError, ValueError):
        return "Enter an output FPS from 1 to 60, or use 0 to match driver timing."
    if not math.isfinite(requested_fps) or requested_fps < 0:
        return "Enter an output FPS from 1 to 60, or use 0 to match driver timing."
    if requested_fps and not 1 <= requested_fps <= 60:
        return "Enter an output FPS from 1 to 60, or use 0 to match driver timing."

    if not driving_video:
        return "Upload or record a driver clip to see its FPS and estimated output duration."

    path = Path(driving_video)
    if not path.is_file():
        return "Driver video metadata is unavailable for this clip."

    driver_fps = _read_driver_fps(path)
    driver_duration = _read_driver_duration(path)
    lines = [
        f"- **Detected driver FPS:** {driver_fps:.3g} FPS"
        if driver_fps is not None
        else "- **Detected driver FPS:** unavailable"
    ]

    if requested_fps > 0:
        selected_fps = requested_fps
        lines.append(f"- **Output FPS:** {selected_fps:.3g} FPS (override)")
    elif driver_fps is not None:
        selected_fps = match_driver_output_fps(driver_fps)
        lines.append(
            f"- **Output FPS:** {selected_fps:.3g} FPS (matched to driver timing)"
        )
    else:
        selected_fps = None
        lines.append(
            "- **Output FPS:** unavailable; set an explicit output FPS to render"
        )

    if driver_duration is None:
        lines.append(
            "- **Estimated output duration:** unavailable because video duration "
            "metadata could not be read."
        )
    elif driver_fps is None or selected_fps is None:
        lines.append(
            "- **Estimated output duration:** unavailable because driver FPS "
            "metadata could not be read."
        )
    else:
        matched_fps = match_driver_output_fps(driver_fps)
        estimated_duration = driver_duration * matched_fps / selected_fps
        lines.append(
            f"- **Estimated output duration:** {estimated_duration:.1f} seconds"
        )
    return "\n".join(lines)


def _prepare_driver_preview_frame(frame, max_edge: int, cv2, np):
    height, width = frame.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale < 1:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


def extract_driver_preview(
    driving_video: str | Path | None, max_edge: int = 960
):
    """Decode one middle frame for visual review without loading render models."""
    if not driving_video:
        return None
    video_path = Path(driving_video)
    if not video_path.is_file():
        return None

    try:
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return None
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            success, frame = capture.read()
            if not success and frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = capture.read()
            if not success or frame is None:
                return None
            return _prepare_driver_preview_frame(frame, max_edge, cv2, np)
        finally:
            capture.release()
    except (ImportError, OSError, ValueError):
        return None


def extract_driver_preview_frames(
    driving_video: str | Path | None,
    max_frames: int = 4,
    max_edge: int = 960,
):
    """Decode evenly spaced driver moments for review without render models."""
    if not driving_video:
        return []
    video_path = Path(driving_video)
    if not video_path.is_file():
        return []
    if max_frames < 1:
        raise ValueError("The driver preview must request at least one frame.")
    if max_edge < 1:
        raise ValueError("The driver preview maximum edge must be positive.")

    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    try:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return []
            raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = (
                max(0, int(raw_frame_count))
                if math.isfinite(raw_frame_count)
                else 0
            )
            raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None

            sample_count = min(frame_count, max_frames) if frame_count else 1
            if sample_count == 1:
                frame_indices = [0]
            else:
                frame_indices = sorted(
                    {
                        round(index * (frame_count - 1) / (sample_count - 1))
                        for index in range(sample_count)
                    }
                )

            previews = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    continue
                preview = _prepare_driver_preview_frame(
                    frame, max_edge, cv2, np
                )
                label = (
                    f"{frame_index / fps:.1f} s"
                    if fps is not None
                    else f"Frame {frame_index + 1}"
                )
                previews.append((preview, label))
            return previews
        finally:
            capture.release()
    except (OSError, ValueError, cv2.error):
        return []


def inspect_full_body_inputs(
    source_image: str | Path | None,
    driving_video: str | Path | None,
) -> FullBodyInputReport:
    """Inspect framing and duration without loading the generation models."""

    target_width = target_height = None
    warnings: list[str] = []
    if source_image:
        target_path = Path(source_image)
        if target_path.is_file():
            target_width, target_height = _read_target_dimensions(target_path)
            if target_width is None or target_height is None:
                warnings.append(
                    "The target image could not be decoded. Re-export it as a "
                    "PNG or JPEG before rendering."
                )
            else:
                aspect_ratio = target_width / target_height
                if aspect_ratio >= 1.0:
                    warnings.append(
                        "The target image is landscape. Use a portrait image "
                        "with the head, feet, and both hands inside the frame; "
                        "cropped body parts cannot be reconstructed."
                    )
                elif aspect_ratio > 0.9:
                    warnings.append(
                        "The target image is close to square. Confirm that the "
                        "head, feet, and both hands are visible with some margin."
                    )
                elif aspect_ratio < 0.35:
                    warnings.append(
                        "The target image is unusually narrow. Use a wider "
                        "full-body frame so both hands and the body stay visible."
                    )
                if min(target_width, target_height) < 512:
                    warnings.append(
                        "The target image is low resolution. Use at least 512 px "
                        "on its shorter side for more stable clothing and hand detail."
                    )
        else:
            warnings.append("The target image path is not available for inspection.")
    else:
        warnings.append("Upload a full-body target image before rendering.")

    driver_duration = None
    if driving_video:
        driver_path = Path(driving_video)
        if driver_path.is_file():
            driver_duration = _read_driver_duration(driver_path)
            if driver_duration is None:
                warnings.append(
                    "The driver video duration could not be read. Re-encode it "
                    "as a playable MP4 before rendering."
                )
            elif driver_duration < 1.0:
                warnings.append(
                    f"The driver video is only {driver_duration:.1f} seconds. "
                    "Record at least 1 second of clear full-body motion."
                )
            elif driver_duration > 30.0:
                warnings.append(
                    f"The driver video is {driver_duration:.1f} seconds long. "
                    "Use a clip under 30 seconds to keep the offline render practical."
                )
        else:
            warnings.append("The driver video path is not available for inspection.")
    else:
        warnings.append("Record or upload a driver video before rendering.")

    return FullBodyInputReport(
        target_width=target_width,
        target_height=target_height,
        driver_duration_seconds=driver_duration,
        warnings=tuple(warnings),
    )


class MimicMotionBackend:
    """Run official MimicMotion in its own Python environment."""

    def __init__(
        self,
        root: Optional[Path] = None,
        python_executable: Optional[Path] = None,
        runner_script: Optional[Path] = None,
        pose_preflight_runner: Optional[Path] = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.root = Path(
            root
            or os.environ.get("MIMICMOTION_ROOT", project_root / "third_party" / "MimicMotion")
        ).expanduser()
        self.python_executable = Path(
            python_executable
            or os.environ.get("MIMICMOTION_PYTHON", self._default_python_path())
        ).expanduser()
        self.runner_script = Path(
            runner_script
            or os.environ.get(
                "MIMICMOTION_RUNNER",
                project_root / "mimicmotion_low_vram_runner.py",
            )
        ).expanduser()
        self.pose_preflight_runner = Path(
            pose_preflight_runner
            or os.environ.get(
                "MIMICMOTION_PREFLIGHT_RUNNER",
                project_root / "scripts" / "mimicmotion_pose_preflight.py",
            )
        ).expanduser()
        self._process_lock = threading.Lock()
        self._active_process: Optional[subprocess.Popen] = None
        self._cancel_requested = False

    def cancel_active_render(self) -> bool:
        """Terminate the active MimicMotion process without touching other modes."""
        with self._process_lock:
            process = self._active_process
            if process is None or process.poll() is not None:
                return False
            try:
                process.terminate()
            except OSError:
                return False
            self._cancel_requested = True
            return True

    def _default_python_path(self) -> str:
        if platform.system().lower() == "windows":
            return str(self.root / ".venv" / "Scripts" / "python.exe")
        return str(self.root / ".venv" / "bin" / "python")

    @property
    def required_files(self) -> tuple[Path, ...]:
        return (
            self.root / "inference.py",
            self.root / "models" / "DWPose" / "yolox_l.onnx",
            self.root / "models" / "DWPose" / "dw-ll_ucoco_384.onnx",
            self.root / "models" / "MimicMotion_1-1.pth",
        )

    def preflight(
        self, source_image: Path, driving_video: Path
    ) -> FullBodyInputReport:
        if not source_image.is_file():
            raise FullBodyAnimationError(
                f"Full-body target image was not found: {source_image}"
            )
        if not driving_video.is_file():
            raise FullBodyAnimationError(
                f"Recorded pose-driving video was not found: {driving_video}"
            )
        if not (self.root / "inference.py").is_file() or not self.runner_script.is_file():
            raise FullBodyAnimationError(
                "MimicMotion is not installed. Run setup_full_body_windows.bat "
                "from the project root on the Alienware."
            )
        if not self.python_executable.is_file():
            raise FullBodyAnimationError(
                f"MimicMotion Python was not found: {self.python_executable}. "
                "Set MIMICMOTION_PYTHON or run setup_full_body_windows.bat."
            )
        missing = [path for path in self.required_files if not path.is_file()]
        if missing:
            names = ", ".join(str(path.relative_to(self.root)) for path in missing)
            raise FullBodyAnimationError(
                "MimicMotion model assets are incomplete. Missing: "
                f"{names}. Run setup_full_body_windows.bat."
            )
        return inspect_full_body_inputs(source_image, driving_video)

    def inspect_driver_video(
        self, driving_video: str | Path | None, max_samples: int = 12
    ) -> tuple[str, ...]:
        """Run sparse DWPose Wholebody checks in the isolated MimicMotion env."""
        if not driving_video:
            return ("Record or upload a driver video before running the pose check.",)
        driving_video = Path(driving_video)
        if not driving_video.is_file():
            return ("The driver video is unavailable for the automatic pose check.",)

        pose_models = (
            self.root / "models" / "DWPose" / "yolox_l.onnx",
            self.root / "models" / "DWPose" / "dw-ll_ucoco_384.onnx",
        )
        missing_models = [path.name for path in pose_models if not path.is_file()]
        if missing_models:
            return (
                "Automatic DWPose checking is unavailable because model assets are "
                f"missing ({', '.join(missing_models)}). Run "
                "setup_full_body_windows.bat, then check the clip manually.",
            )
        if not self.python_executable.is_file():
            return (
                "Automatic DWPose checking is unavailable because the isolated "
                "MimicMotion Python environment is missing. Run "
                "setup_full_body_windows.bat, then check the clip manually.",
            )
        if not self.pose_preflight_runner.is_file():
            return (
                "Automatic DWPose checking is unavailable because its preflight "
                "runner is missing. Update the project files and check the clip manually.",
            )

        command = [
            str(self.python_executable),
            str(self.pose_preflight_runner),
            "--inspect-driver",
            str(driving_video.resolve()),
            "--max-samples",
            str(max(1, min(int(max_samples), 12))),
        ]
        started_at = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=DWPOSE_PREFLIGHT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial_stderr = exc.stderr or ""
            if isinstance(partial_stderr, bytes):
                partial_stderr = partial_stderr.decode("utf-8", errors="replace")
            last_progress = next(
                (
                    line.strip()
                    for line in reversed(partial_stderr.splitlines())
                    if line.strip().startswith("DWPose progress:")
                ),
                None,
            )
            progress_note = (
                f" Last reported status: {last_progress}."
                if last_progress
                else ""
            )
            return (
                "The automatic DWPose check timed out after "
                f"{DWPOSE_PREFLIGHT_TIMEOUT_SECONDS} seconds.{progress_note} "
                "Use a shorter, clearly visible single-person clip and review "
                "it before rendering.",
            )
        except OSError as exc:
            elapsed = time.perf_counter() - started_at
            return (
                "The automatic DWPose check could not start or finish "
                f"after {elapsed:.2f} seconds ({exc}). Check the isolated "
                "MimicMotion environment and review the clip manually.",
            )

        elapsed = time.perf_counter() - started_at

        if result.returncode != 0:
            detail_lines = (result.stderr or result.stdout).strip().splitlines()
            detail = detail_lines[-1] if detail_lines else "No diagnostic was returned."
            return (
                f"The automatic DWPose check failed after {elapsed:.2f} seconds: "
                f"{detail[:240]} Check the driver manually before rendering.",
            )
        try:
            json_lines = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().startswith("{")
            ]
            payload = json.loads(json_lines[-1]) if json_lines else {}
        except (json.JSONDecodeError, TypeError):
            return (
                f"The automatic DWPose check returned unreadable results after "
                f"{elapsed:.2f} seconds. "
                "Check the driver manually before rendering.",
            )
        sampled_frames = payload.get("frames")
        sample_count = (
            len(sampled_frames)
            if isinstance(sampled_frames, list)
            else payload.get("sample_count", 0)
        )
        warnings = format_dwp_pose_warnings(payload)
        timing = (
            f"DWPose preflight elapsed: {elapsed:.2f} seconds for "
            f"{sample_count} sampled frames, including isolated Python/model startup."
        )
        return (*warnings, timing)

    @staticmethod
    def _yaml_string(value: Path | str) -> str:
        return json.dumps(str(value.resolve()).replace("\\", "/"))

    def _write_config(
        self,
        config_path: Path,
        source_image: Path,
        driving_video: Path,
        options: FullBodyRenderOptions,
    ) -> None:
        options.validate()
        config_path.write_text(
            "\n".join(
                [
                    "base_model_path: stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                    f"ckpt_path: {self._yaml_string(self.root / 'models' / 'MimicMotion_1-1.pth')}",
                    "test_case:",
                    f"  - ref_video_path: {self._yaml_string(driving_video)}",
                    f"    ref_image_path: {self._yaml_string(source_image)}",
                    f"    num_frames: {options.chunk_size}",
                    f"    resolution: {options.resolution}",
                    f"    frames_overlap: {options.frames_overlap}",
                    f"    num_inference_steps: {options.num_inference_steps}",
                    f"    noise_aug_strength: {options.noise_aug_strength}",
                    f"    guidance_scale: {options.guidance_scale}",
                    f"    sample_stride: {options.sample_stride}",
                    f"    fps: {options.fps}",
                    f"    seed: {options.seed}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def render(
        self,
        source_image: Path,
        driving_video: Path,
        output_dir: Path,
        options: FullBodyRenderOptions = FullBodyRenderOptions(),
    ) -> Path:
        """Render target-owned body, clothing, hands, and face offline."""

        source_image = Path(source_image).resolve()
        driving_video = Path(driving_video).resolve()
        output_dir = Path(output_dir).resolve()
        self.preflight(source_image, driving_video)
        options.validate()
        render_options = options
        if options.fps is None:
            driver_fps = _read_driver_fps(driving_video)
            if driver_fps is None:
                raise FullBodyAnimationError(
                    "Could not read the driver video's frame rate. Set an explicit "
                    "output FPS and retry, or re-encode the driver as a playable MP4."
                )
            render_options = replace(
                options,
                # MimicMotion's get_video_pose applies its own FPS-based stride.
                # Keep the configured base stride unchanged to avoid sampling twice.
                fps=match_driver_output_fps(driver_fps, options.sample_stride),
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "mimicmotion_full_body.yaml"
        self._write_config(config_path, source_image, driving_video, render_options)
        attempt_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        log_path = output_dir / f"mimicmotion-{attempt_id}.log"
        command = [
            str(self.python_executable),
            str(self.runner_script),
            "--inference_config",
            str(config_path),
            "--output_dir",
            str(output_dir),
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

        started_at = time.time()
        with self._process_lock:
            if self._active_process is not None:
                raise FullBodyAnimationError("A full-body render is already running.")
            self._cancel_requested = False
            log_file = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log_file.close()
                raise
            self._active_process = process
        try:
            return_code = process.wait()
        finally:
            log_file.close()
            with self._process_lock:
                was_cancelled = (
                    self._active_process is process and self._cancel_requested
                )
                if self._active_process is process:
                    self._active_process = None
                self._cancel_requested = False

        if was_cancelled:
            raise FullBodyAnimationCancelled(
                "Full-body render cancelled. This attempt's log is preserved at "
                f"{log_path}. Existing and partial output files remain in {output_dir}.",
                log_path=log_path,
            )
        if return_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            if any("PyAV is not installed" in line for line in tail):
                raise FullBodyAnimationError(
                    "MimicMotion completed inference but could not write the MP4 "
                    "because PyAV is missing. Install it in the isolated "
                    "MimicMotion environment with "
                    "`third_party\\MimicMotion\\.venv\\Scripts\\python.exe "
                    "-m pip install av==12.0.0`, then rerun the render.\n\n"
                    f"See {log_path}."
                )
            raise FullBodyAnimationError(
                "MimicMotion failed. See "
                f"{log_path}.\n\n" + "\n".join(tail)
            )

        outputs = [
            path
            for path in output_dir.glob("*.mp4")
            if path.stat().st_mtime >= started_at
        ]
        if not outputs:
            raise FullBodyAnimationError(
                f"MimicMotion finished without producing an MP4. See {log_path}."
            )
        return max(outputs, key=lambda path: path.stat().st_mtime)


class FullBodyAnimationPipeline:
    """Application-facing full-body renderer with isolated backend discovery."""

    def __init__(self, backend: Optional[MimicMotionBackend] = None):
        self.backend = backend or MimicMotionBackend()

    def cancel_active_render(self) -> bool:
        """Request cancellation of this pipeline's active offline render."""
        return self.backend.cancel_active_render()

    def inspect_driver_video(
        self, driving_video: str | Path | None, max_samples: int = 12
    ) -> tuple[str, ...]:
        return self.backend.inspect_driver_video(driving_video, max_samples)

    def render(
        self,
        source_image: str | Path,
        driving_video: str | Path,
        output_dir: str | Path = "results/full_body",
        options: FullBodyRenderOptions = FullBodyRenderOptions(),
    ) -> Path:
        return self.backend.render(
            Path(source_image),
            Path(driving_video),
            Path(output_dir),
            options,
        )