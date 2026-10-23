"""Local subject cutout and webcam-background compositing for portrait renders."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SF3D_ROOT = PROJECT_ROOT / "third_party" / "stable-fast-3d"
BACKGROUND_REMOVAL_WORKER = PROJECT_ROOT / "scripts" / "remove_photo_background.py"


class DriverBackgroundError(RuntimeError):
    """The target portrait could not be isolated for background compositing."""


def _photo_model_python() -> Path:
    """Use rembg from the isolated photo-to-3D environment to avoid app conflicts."""
    if os.name == "nt":
        return SF3D_ROOT / ".venv" / "Scripts" / "python.exe"
    return SF3D_ROOT / ".venv" / "bin" / "python"


def extract_target_alpha(source_image: str | Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Remove the still-photo background once and return a feathered target alpha."""
    image_path = Path(source_image).expanduser().resolve()
    if not image_path.is_file():
        raise DriverBackgroundError("The target photo is no longer available on this computer.")

    python = _photo_model_python()
    if not python.is_file():
        raise DriverBackgroundError(
            "Webcam-background mode needs the local Stable Fast 3D environment, "
            "which provides background removal. Set up the local photo-to-3D "
            "environment first; your photo remains on this computer."
        )
    if not BACKGROUND_REMOVAL_WORKER.is_file():
        raise DriverBackgroundError("The local background-removal worker is missing.")

    with tempfile.TemporaryDirectory(prefix="portrait-background-") as temp_dir:
        output_path = Path(temp_dir) / "target_rgba.png"
        try:
            result = subprocess.run(
                [str(python), str(BACKGROUND_REMOVAL_WORKER), str(image_path), str(output_path)],
                cwd=SF3D_ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DriverBackgroundError(
                "Local background removal took longer than 3 minutes and was stopped."
            ) from exc
        if result.returncode != 0 or not output_path.is_file():
            detail = (result.stderr or result.stdout or "No diagnostic output was returned.")
            detail = "\n".join(detail.strip().splitlines()[-8:])
            raise DriverBackgroundError(
                "Could not remove the target photo background locally. "
                "Check that rembg is installed in the Stable Fast 3D environment.\n"
                + detail
            )

        rgba = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise DriverBackgroundError(
                "The local background-removal model did not return an alpha mask."
            )
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    height, width = target_shape
    if height < 1 or width < 1:
        raise DriverBackgroundError("The target portrait has invalid dimensions.")
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

    # Keep the model's soft edges while avoiding a visibly jagged cutout.
    sigma = max(0.6, min(height, width) * 0.0015)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(alpha, 0.0, 1.0)


def _cover_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize and center-crop a driver frame without stretching its contents."""
    source_height, source_width = image.shape[:2]
    if source_height < 1 or source_width < 1:
        raise ValueError("The webcam frame is empty.")
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, int(round(source_width * scale)))
    resized_height = max(height, int(round(source_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return resized[top : top + height, left : left + width]


def composite_target_over_driver_background(
    target_rgb: np.ndarray,
    driver_bgr: np.ndarray,
    target_alpha: np.ndarray,
) -> np.ndarray:
    """Place the target-person cutout over the current webcam frame; return RGB."""
    if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
        raise ValueError("The target portrait must be an RGB image.")
    if driver_bgr.ndim != 3 or driver_bgr.shape[2] != 3:
        raise ValueError("The driver frame must be a BGR image.")

    height, width = target_rgb.shape[:2]
    background_rgb = cv2.cvtColor(_cover_resize(driver_bgr, height, width), cv2.COLOR_BGR2RGB)
    alpha = np.asarray(target_alpha, dtype=np.float32)
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    composed = (
        target_rgb.astype(np.float32) * alpha
        + background_rgb.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(composed, 0, 255).astype(np.uint8)
