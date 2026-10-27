"""OpenCV camera and approximate 3D stick-figure windows for skeletal mode."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .skeletal_frame import BODY_CONNECTIONS, SkeletalFrame


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _draw_label(image, text: str, position: tuple[int, int], color=(245, 245, 245)):
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (10, 10, 10), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        color, 1, cv2.LINE_AA,
    )


def render_camera_view(
    frame_bgr: np.ndarray,
    skeletal: Optional[SkeletalFrame],
    status: str,
    inference_ms: float = 0.0,
    dropped_frames: int = 0,
) -> np.ndarray:
    """Mirror the display and draw confidence-qualified body/hand points."""
    view = cv2.flip(frame_bgr, 1)
    height, width = view.shape[:2]
    if skeletal is not None:
        points = {
            name: (int((1.0 - x) * width), int(y * height))
            for name, (x, y) in skeletal.image_landmarks.items()
        }
        for start, end in BODY_CONNECTIONS:
            if start in points and end in points:
                cv2.line(view, points[start], points[end], (64, 210, 120), 2, cv2.LINE_AA)
        for side in ("left", "right"):
            hand_points = {}
            for index in range(21):
                name = f"{side}_hand_{index:02d}"
                if name in points:
                    hand_points[index] = points[name]
            for start, end in HAND_CONNECTIONS:
                if start in hand_points and end in hand_points:
                    cv2.line(
                        view, hand_points[start], hand_points[end],
                        (70, 170, 255), 1, cv2.LINE_AA,
                    )
        for name, point in points.items():
            if not name.startswith("face_"):
                cv2.circle(view, point, 3, (245, 245, 245), -1, cv2.LINE_AA)

    _draw_label(view, "Native skeletal tracking | C calibrate | Q quit", (12, 24))
    _draw_label(view, status, (12, 50), (110, 240, 255))
    if skeletal is not None:
        depth = (
            "relative scale estimate"
            if skeletal.calibrated
            else "uncalibrated depth"
        )
        _draw_label(
            view,
            f"2.5D | {depth} | inference {inference_ms:.1f} ms | dropped {dropped_frames}",
            (12, 76),
        )
    return view


def render_skeleton_view(
    skeletal: Optional[SkeletalFrame],
    status: str,
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    """Project relative 2.5D landmarks into an OpenCV stick-figure viewport."""
    canvas = np.full((height, width, 3), (20, 24, 31), dtype=np.uint8)
    center_x = width // 2
    floor_y = int(height * 0.78)
    cv2.rectangle(canvas, (0, 0), (width, 94), (28, 33, 43), -1)
    _draw_label(canvas, "Skeletal preview — approximate monocular 2.5D", (18, 28))
    _draw_label(canvas, status, (18, 55), (110, 240, 255))
    _draw_label(canvas, "Depth is estimated from body scale, not measured.", (18, 80))

    # Simple perspective floor grid. The floor is a visual reference, not a
    # calibrated physical plane.
    for offset in range(-4, 5):
        x = center_x + offset * 46
        cv2.line(canvas, (x, floor_y), (center_x + offset * 94, height - 1), (43, 49, 60), 1)
    for y in (floor_y + 20, floor_y + 52, floor_y + 98, floor_y + 160):
        if y < height:
            cv2.line(canvas, (0, y), (width - 1, y), (43, 49, 60), 1)

    if skeletal is None or not skeletal.landmarks:
        _draw_label(canvas, "Waiting for a full-body pose...", (160, height // 2))
        return canvas

    # Use a fixed virtual-camera distance and let the estimated root depth
    # affect perspective. All values stay in normalized body-height units.
    root_x, root_y, root_depth = skeletal.root_translation
    camera_distance = max(1.1, 2.0 + root_depth)
    focal_length = min(width, height) * 0.72
    projected: dict[str, tuple[int, int]] = {}
    for name, (x, y, z) in skeletal.landmarks.items():
        depth = max(0.55, camera_distance + z)
        scale = focal_length / depth
        px = center_x - int((float(x) + root_x) * scale)  # selfie mirror
        py = floor_y - int((float(y) + root_y) * scale)
        if -width <= px < 2 * width and -height <= py < 2 * height:
            projected[name] = (px, py)

    for start, end in BODY_CONNECTIONS:
        if start in projected and end in projected:
            cv2.line(canvas, projected[start], projected[end], (80, 220, 145), 4, cv2.LINE_AA)
    for side in ("left", "right"):
        hand_points = {
            index: projected[f"{side}_hand_{index:02d}"]
            for index in range(21)
            if f"{side}_hand_{index:02d}" in projected
        }
        for start, end in HAND_CONNECTIONS:
            if start in hand_points and end in hand_points:
                cv2.line(canvas, hand_points[start], hand_points[end], (75, 175, 255), 2, cv2.LINE_AA)
    for name, point in projected.items():
        radius = 6 if name in ("left_ankle", "right_ankle", "left_wrist", "right_wrist") else 4
        cv2.circle(canvas, point, radius, (235, 245, 250), -1, cv2.LINE_AA)

    for side, locked in skeletal.foot_contacts.items():
        if locked and f"{side}_ankle" in projected:
            cv2.circle(canvas, projected[f"{side}_ankle"], 10, (80, 210, 255), 2, cv2.LINE_AA)

    if not skeletal.calibrated:
        _draw_label(canvas, "Press C in a neutral, full-body pose to calibrate.", (22, height - 22))
    else:
        x, y, z = skeletal.root_translation
        _draw_label(canvas, f"Root offset (body heights): X {x:+.2f}  Y {y:+.2f}  Z~ {z:+.2f}",
                    (22, height - 22))
    return canvas


class LiveSkeletalPreview:
    """Own the two OpenCV windows and translate keyboard input to actions."""

    CAMERA_WINDOW = "Your camera — skeletal mode"
    SKELETON_WINDOW = "Approximate 3D skeleton"

    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        cv2.namedWindow(self.CAMERA_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.CAMERA_WINDOW, 720, 600)
        cv2.moveWindow(self.CAMERA_WINDOW, 20, 40)
        cv2.namedWindow(self.SKELETON_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.SKELETON_WINDOW, 640, 640)
        cv2.moveWindow(self.SKELETON_WINDOW, 760, 40)

    def show(
        self,
        camera_frame: np.ndarray,
        skeletal: Optional[SkeletalFrame],
        status: str,
        inference_ms: float = 0.0,
        dropped_frames: int = 0,
    ) -> None:
        cv2.imshow(
            self.CAMERA_WINDOW,
            render_camera_view(camera_frame, skeletal, status, inference_ms, dropped_frames),
        )
        cv2.imshow(
            self.SKELETON_WINDOW,
            render_skeleton_view(skeletal, status),
        )

    def poll(self) -> Optional[str]:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key in (ord("c"), ord("C")):
            return "calibrate"
        for window in (self.CAMERA_WINDOW, self.SKELETON_WINDOW):
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit"
            except cv2.error:
                return "quit"
        return None

    def close(self) -> None:
        cv2.destroyWindow(self.CAMERA_WINDOW)
        cv2.destroyWindow(self.SKELETON_WINDOW)
