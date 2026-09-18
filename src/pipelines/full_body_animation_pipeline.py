"""Offline full-body avatar animation through an isolated MimicMotion runtime.

The regular FasterLivePortrait pipeline is intentionally face-only.  This
adapter keeps the body animation dependency isolated because MimicMotion uses
its own PyTorch/diffusers environment and model checkpoints.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class FullBodyAnimationError(RuntimeError):
    """A user-actionable full-body rendering error."""


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
    fps: int = 15
    seed: int = 42

    def validate(self) -> None:
        if self.resolution not in (512, 576):
            raise ValueError("Full-body resolution must be 512 or 576.")
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
        if not 1 <= self.fps <= 60:
            raise ValueError("Output FPS must be between 1 and 60.")


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
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "mimicmotion_full_body.yaml"
        self._write_config(config_path, source_image, driving_video, options)
        log_path = output_dir / "mimicmotion.log"
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
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=self.root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
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