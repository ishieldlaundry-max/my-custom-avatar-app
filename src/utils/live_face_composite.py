"""Face-only compositing for the live webcam identity-swap path."""

from __future__ import annotations

import cv2
import numpy as np


def build_face_mask(
    landmarks: np.ndarray,
    crop_shape: tuple[int, ...],
    expansion_ratio: float = 0.06,
    feather_ratio: float = 0.05,
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


def map_landmarks_to_crop_coordinates(
    landmarks_in_image: np.ndarray,
    crop_to_image: np.ndarray,
) -> np.ndarray:
    """Map face landmarks from the original image into the model crop."""
    points = np.asarray(landmarks_in_image, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Face landmarks must be an Nx2 coordinate array.")

    matrix = np.asarray(crop_to_image, dtype=np.float32)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("The crop-to-image transform must be a finite 2x3 or 3x3 matrix.")
    try:
        image_to_crop = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The crop-to-image transform must be invertible.") from exc

    points_h = np.concatenate(
        (points[:, :2], np.ones((len(points), 1), dtype=np.float32)),
        axis=1,
    )
    mapped = points_h @ image_to_crop.T
    if np.any(np.abs(mapped[:, 2]) < 1e-8):
        raise ValueError("The image-to-crop transform produced invalid landmarks.")
    return mapped[:, :2] / mapped[:, 2:3]


def build_face_mask_from_image_landmarks(
    landmarks_in_image: np.ndarray,
    crop_to_image: np.ndarray,
    crop_shape: tuple[int, ...],
) -> np.ndarray:
    """Build the alpha mask after converting image-space landmarks to crop space."""
    landmarks_in_crop = map_landmarks_to_crop_coordinates(
        landmarks_in_image,
        crop_to_image,
    )
    return build_face_mask(landmarks_in_crop, crop_shape)


def _match_face_crop_color(
    generated_rgb: np.ndarray,
    driver_crop_rgb: np.ndarray,
    alpha: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply a restrained per-channel correction using downsampled face statistics."""
    sample_size = (min(128, generated_rgb.shape[1]), min(128, generated_rgb.shape[0]))
    generated_small = cv2.resize(
        generated_rgb, sample_size, interpolation=cv2.INTER_AREA
    )
    driver_small = cv2.resize(
        driver_crop_rgb, sample_size, interpolation=cv2.INTER_AREA
    )
    alpha_small = cv2.resize(
        alpha, sample_size, interpolation=cv2.INTER_AREA
    )
    sample = alpha_small > 0.15
    if int(sample.sum()) < 16:
        return generated_rgb.copy()

    generated_values = generated_small[sample].astype(np.float32)
    driver_values = driver_small[sample].astype(np.float32)
    generated_mean = generated_values.mean(axis=0)
    driver_mean = driver_values.mean(axis=0)
    generated_std = np.maximum(generated_values.std(axis=0), 1.0)
    driver_std = np.maximum(driver_values.std(axis=0), 1.0)
    scale = np.clip(driver_std / generated_std, 0.75, 1.35)
    offset = driver_mean - generated_mean * scale
    color_transform = np.column_stack((np.diag(scale), offset)).astype(np.float32)
    matched = cv2.transform(generated_rgb, color_transform)
    return cv2.addWeighted(
        generated_rgb, 1.0 - strength, matched, strength, 0.0
    )


def composite_photo_face_on_driver(
    driver_bgr: np.ndarray,
    generated_face_rgb: np.ndarray,
    face_mask: np.ndarray,
    crop_to_driver: np.ndarray,
    color_match_strength: float = 0.35,
) -> np.ndarray:
    """Place an animated photo-face crop on the live frame, preserving its body/background."""
    if driver_bgr.ndim != 3 or driver_bgr.shape[2] != 3:
        raise ValueError("The live driver frame must be a BGR color image.")
    if generated_face_rgb.ndim != 3 or generated_face_rgb.shape[2] != 3:
        raise ValueError("The generated face must be an RGB color image.")

    transform = np.asarray(crop_to_driver, dtype=np.float32)
    if transform.shape == (2, 3):
        transform = np.vstack(
            (transform, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
        )
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError(
            "The face-crop-to-camera transform must be a finite 2x3 or 3x3 matrix."
        )
    if not np.isfinite(color_match_strength) or not 0.0 <= color_match_strength <= 1.0:
        raise ValueError("Color-match strength must be between 0 and 1.")
    matrix = transform[:2]

    alpha = np.asarray(face_mask, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.ndim != 2 or min(alpha.shape) < 1:
        raise ValueError("The face mask must be a non-empty 2-D array.")
    if alpha.shape != generated_face_rgb.shape[:2]:
        # M_c2o maps coordinates from the 512x512 driver crop to the camera
        # frame, while LivePortrait's network output is 256x256. Keep the mask
        # and affine transform in the same crop coordinate space by upscaling
        # the generated face instead of shrinking the mask to the network size.
        generated_face_rgb = cv2.resize(
            generated_face_rgb,
            (alpha.shape[1], alpha.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    if alpha.max(initial=0) > 1.0:
        alpha = alpha / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)

    crop_height, crop_width = generated_face_rgb.shape[:2]
    if color_match_strength:
        try:
            driver_to_crop = np.linalg.inv(transform)[:2]
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The face-crop-to-camera transform must be invertible."
            ) from exc
        driver_crop_rgb = cv2.warpAffine(
            cv2.cvtColor(driver_bgr, cv2.COLOR_BGR2RGB),
            driver_to_crop,
            (crop_width, crop_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        generated_face_rgb = _match_face_crop_color(
            generated_face_rgb,
            driver_crop_rgb,
            alpha,
            color_match_strength,
        )

    height, width = driver_bgr.shape[:2]
    face_bgr = cv2.cvtColor(generated_face_rgb, cv2.COLOR_RGB2BGR)
    crop_rows, crop_columns = np.where(alpha > 0.001)
    if len(crop_rows) == 0:
        return driver_bgr.copy()
    crop_corners = np.array(
        [
            [crop_columns.min(), crop_rows.min()],
            [crop_columns.max(), crop_rows.min()],
            [crop_columns.max(), crop_rows.max()],
            [crop_columns.min(), crop_rows.max()],
        ],
        dtype=np.float32,
    )
    driver_corners = crop_corners @ matrix[:, :2].T + matrix[:, 2]
    left = max(0, int(np.floor(driver_corners[:, 0].min())) - 2)
    top = max(0, int(np.floor(driver_corners[:, 1].min())) - 2)
    right = min(width, int(np.ceil(driver_corners[:, 0].max())) + 3)
    bottom = min(height, int(np.ceil(driver_corners[:, 1].max())) + 3)
    if left >= right or top >= bottom:
        return driver_bgr.copy()

    local_matrix = matrix.copy()
    local_matrix[:, 2] -= np.array([left, top], dtype=np.float32)
    size = (right - left, bottom - top)
    warped_face = cv2.warpAffine(
        face_bgr,
        local_matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpAffine(
        alpha,
        local_matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)[..., None]
    driver_roi = driver_bgr[top:bottom, left:right].astype(np.float32)
    result = driver_bgr.copy()
    result[top:bottom, left:right] = (
        warped_face.astype(np.float32) * warped_alpha
        + driver_roi * (1.0 - warped_alpha)
    )
    return result
