"""Optional face and hand detail sheets for full-body benchmark videos."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SAMPLE_LOCATIONS = (("15%", 0.15), ("50%", 0.5), ("85%", 0.85))
REGIONS = (
    ("Face", "face", 20, 1.9),
    ("Hand 1", "hand_1", 7, 2.4),
    ("Hand 2", "hand_2", 7, 2.4),
)
CONFIDENCE_THRESHOLD = 0.3
CELL_WIDTH = 320
CELL_HEIGHT = 296


def sample_frame_indices(frame_count: int) -> list[tuple[str, int]]:
    if frame_count < 1:
        return []
    return [
        (label, min(frame_count - 1, round((frame_count - 1) * fraction)))
        for label, fraction in SAMPLE_LOCATIONS
    ]


def _landmark_crop(
    image: Image.Image,
    region: dict,
    min_points: int,
    scale: float,
) -> Image.Image | None:
    points = region.get("points")
    scores = region.get("scores")
    if not isinstance(points, list) or not isinstance(scores, list):
        return None
    if len(points) != len(scores):
        return None

    valid = []
    for point, score in zip(points, scores):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x, y, confidence = float(point[0]), float(point[1]), float(score)
        except (TypeError, ValueError):
            continue
        if (
            not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
            or not math.isfinite(confidence)
            or confidence < CONFIDENCE_THRESHOLD
        ):
            continue
        valid.append((x * image.width, y * image.height))

    if len(valid) < min_points:
        return None

    xs = [point[0] for point in valid]
    ys = [point[1] for point in valid]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    landmark_size = max(width, height)
    if landmark_size < 3:
        return None

    side = max(landmark_size * scale, 24.0)
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    left = max(0, round(center_x - side / 2))
    top = max(0, round(center_y - side / 2))
    right = min(image.width, round(center_x + side / 2))
    bottom = min(image.height, round(center_y + side / 2))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def make_detail_sheet(
    sample_frames: list[tuple[str, int, Image.Image]],
    detections: dict,
    preview_path: Path,
) -> tuple[str | None, int]:
    frames_by_index = {
        int(item.get("frame_index", -1)): item
        for item in detections.get("frames", [])
        if isinstance(item, dict)
    }
    sheet = Image.new(
        "RGB",
        (CELL_WIDTH * len(sample_frames), CELL_HEIGHT * len(REGIONS)),
        color=(12, 18, 28),
    )
    draw = ImageDraw.Draw(sheet)
    crop_count = 0

    for column, (timeline_label, frame_index, frame) in enumerate(sample_frames):
        detection = frames_by_index.get(frame_index, {})
        person_count = detection.get("people_count", 0)
        for row, (region_label, key, min_points, scale) in enumerate(REGIONS):
            x = column * CELL_WIDTH
            y = row * CELL_HEIGHT
            draw.text(
                (x + 12, y + 9),
                f"{region_label} - {timeline_label}",
                fill=(235, 245, 255),
            )
            crop = None
            if person_count == 1:
                crop = _landmark_crop(
                    frame,
                    detection.get(key, {}),
                    min_points=min_points,
                    scale=scale,
                )
            if crop is None:
                draw.text(
                    (x + 12, y + 138),
                    "Not reliably detected",
                    fill=(170, 184, 201),
                )
                continue
            crop = ImageOps.pad(
                crop,
                (CELL_WIDTH - 32, CELL_HEIGHT - 58),
                method=Image.Resampling.LANCZOS,
                color=(12, 18, 28),
            )
            sheet.paste(crop, (x + 16, y + 42))
            crop_count += 1

    if crop_count == 0:
        return None, 0
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path)
    return str(preview_path), crop_count


def _backend_python(backend_root: Path) -> Path | None:
    for relative_path in (
        Path(".venv") / "Scripts" / "python.exe",
        Path(".venv") / "bin" / "python",
    ):
        candidate = backend_root / relative_path
        if candidate.is_file():
            return candidate
    return None


def _run_landmark_detector(
    video_path: Path,
    frame_indices: list[int],
    backend_root: Path,
) -> tuple[dict | None, str | None]:
    python = _backend_python(backend_root)
    helper = Path(__file__).resolve().parent / "scripts" / "mimicmotion_detail_landmarks.py"
    if python is None:
        return None, "MimicMotion's isolated Python environment was not found."
    if not helper.is_file():
        return None, "The DWPose detail-detection helper was not found."

    command = [
        str(python),
        str(helper),
        "--video",
        str(video_path.resolve()),
        "--backend-root",
        str(backend_root.resolve()),
        "--frame-indices",
        ",".join(str(index) for index in frame_indices),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(backend_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"DWPose detail detection could not run: {exc}"

    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        message = detail[-1] if detail else f"process exited with {result.returncode}"
        return None, f"DWPose detail detection failed: {message[:240]}"

    for line in reversed(result.stdout.splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            return payload, None
    return None, "DWPose detail detection returned no readable landmark data."


def make_quality_detail_preview(
    video_path: Path,
    preview_path: Path,
    backend_root: Path | None = None,
) -> tuple[str | None, str]:
    """Save enlarged crops only for regions with enough confident landmarks."""
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return None, "The rendered video could not be opened for detail sampling."
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            samples = sample_frame_indices(frame_count)
            if not samples:
                return None, "The rendered video contains no decodable frames."
            sample_frames = []
            for label, frame_index in samples:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sample_frames.append(
                    (label, frame_index, Image.fromarray(rgb))
                )
        finally:
            capture.release()
    except (ImportError, OSError, ValueError) as exc:
        return None, f"Detail sample frames could not be read: {exc}"

    if not sample_frames:
        return None, "No detail sample frames could be decoded."

    backend_root = backend_root or (
        Path(__file__).resolve().parent / "third_party" / "MimicMotion"
    )
    payload, error = _run_landmark_detector(
        video_path,
        [index for _, index, _ in sample_frames],
        backend_root,
    )
    if payload is None:
        return None, error or "DWPose did not return landmarks."

    path, crop_count = make_detail_sheet(sample_frames, payload, preview_path)
    if path is None:
        return None, (
            "No sampled frame had enough confident face or hand landmarks; "
            "detail crops were omitted."
        )
    return path, (
        f"Saved {crop_count} enlarged face/hand crops. Crops are visual samples, "
        "not a quality score."
    )
