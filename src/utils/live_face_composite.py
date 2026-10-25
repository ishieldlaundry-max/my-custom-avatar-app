"""Face-only compositing for the live webcam identity-swap path."""

from __future__ import annotations

import cv2
import numpy as np


def build_face_mask(
    landmarks: np.ndarray,
    crop_shape: tuple[int, ...],
    expansion_ratio: float = 0.025,
    feather_ratio: float = 0.02,
) -> np.ndarray:
    """Build a soft face-oval alpha mask from face landmarks in crop coordinates."""
    height, width = crop_shape[:2]
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Face landmarks must be an Nx2 coordinate array.")
    points = points[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("At least three valid face landmarks are required.")
    if height < 1 or width < 1:
        raise ValueError("The face crop dimensions must be positive.")

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    face_width = max(1, int(np.ptp(hull[:, 0, 0])))
    expansion = max(1, int(round(face_width * expansion_ratio)))
    kernel_size = expansion * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.dilate(mask, kernel)
    sigma = max(1.0, face_width * feather_ratio)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return mask.astype(np.float32) / 255.0


def composite_photo_face_on_driver(
    driver_bgr: np.ndarray,
    generated_face_rgb: np.ndarray,
    face_mask: np.ndarray,
    crop_to_driver: np.ndarray,
) -> np.ndarray:
    """Place an animated photo-face crop on the live frame, preserving its body/background."""
    if driver_bgr.ndim != 3 or driver_bgr.shape[2] != 3:
        raise ValueError("The live driver frame must be a BGR color image.")
    if generated_face_rgb.ndim != 3 or generated_face_rgb.shape[2] != 3:
        raise ValueError("The generated face must be an RGB color image.")

    matrix = np.asarray(crop_to_driver, dtype=np.float32)
    if matrix.shape == (3, 3):
        matrix = matrix[:2]
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError(
            "The face-crop-to-camera transform must be a finite 2x3 or 3x3 matrix."
        )

    crop_height, crop_width = generated_face_rgb.shape[:2]
    alpha = np.asarray(face_mask, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.shape != (crop_height, crop_width):
        alpha = cv2.resize(
            alpha, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR
        )
    if alpha.max(initial=0) > 1.0:
        alpha = alpha / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)

    height, width = driver_bgr.shape[:2]
    size = (width, height)
    face_bgr = cv2.cvtColor(generated_face_rgb, cv2.COLOR_RGB2BGR)
    warped_face = cv2.warpAffine(
        face_bgr,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)[..., None]
    result = (
        warped_face.astype(np.float32) * warped_alpha
        + driver_bgr.astype(np.float32) * (1.0 - warped_alpha)
    )
    return np.clip(result, 0, 255).astype(np.uint8)
