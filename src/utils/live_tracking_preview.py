"""OpenCV diagnostics for the exact camera frames supplied to portrait inference."""
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np

from .driver_background import DriverBackgroundError, median_clean_background


def resolve_camera_source(source):
    """Numeric inputs are camera indices; missing files must not select a camera."""
    text = str(source) if source is not None else ""
    if not text:
        return True, 0
    if text.isdecimal():
        return True, int(text)
    if not Path(text).is_file():
        raise FileNotFoundError(f"Driving video does not exist: {text}")
    return False, text


class LiveTrackingPreview:
    def __init__(
        self,
        camera_index,
        allow_background_capture=False,
        face_swap_mode=False,
    ):
        self.camera_index = camera_index
        self.allow_background_capture = allow_background_capture
        self.face_swap_mode = face_swap_mode
        self.background_capture_requested = False
        self.background_captured = False
        self.needs_reference = True
        self.calibration_requested = False
        self.head_enabled = True
        self.last_key = -1
        self.effects = None
        self.camera_height = 0
        self.buttons = []
        self.pending_actions = deque(maxlen=12)
        self.status = (
            "Camera body/background stay live; photo face is animated. Press C to calibrate."
            if face_swap_mode
            else "Hold a neutral face, eyes open. Press C to calibrate."
        )
        for title, x in (("Your camera", 20), ("Render", 690)):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640, 560)
            cv2.moveWindow(title, x, 60)
        cv2.setMouseCallback("Your camera", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (left, top, right, bottom), key, _ in self.buttons:
            if left <= x < right and top <= y < bottom:
                self.pending_actions.append(key)
                return
        # The toolbar must never be interpreted as a shoulder selection.
        if self.effects is not None and 0 <= x < 640 and 0 <= y < self.camera_height:
            self.effects._driver_click(event, x, y, flags, param)

    def add_controls(self, preview):
        """Draw buttons in image coordinates, which OpenCV maps on resize."""
        self.camera_height = preview.shape[0]
        canvas = np.zeros((self.camera_height + 132, 640, 3), dtype=np.uint8)
        canvas[:self.camera_height, :preview.shape[1]] = preview
        canvas[self.camera_height:] = (30, 26, 23)
        if self.face_swap_mode:
            labels = [
                (ord("c"), "Calibrate face (C)"),
                (ord("h"), f"Head: {'ON' if self.head_enabled else 'OFF'} (H)"),
                (ord("q"), "Stop tracking (Q)"),
            ]
        else:
            labels = [
                (ord("c"), "Calibrate (C)"),
                (ord("h"), f"Head: {'ON' if self.head_enabled else 'OFF'} (H)"),
                (ord("s"), "Set shoulders (S)"),
                (ord("b"), "Shoulders off (B)"),
                (ord("l"), f"Shading: {'ON' if self.effects and self.effects.shadows_enabled else 'OFF'} (L)"),
                (ord("q"), "Stop tracking (Q)"),
            ]
        if self.allow_background_capture:
            labels.insert(
                5,
                (
                    ord("g"),
                    "Retake room (G)" if self.background_captured else "Capture room (G)",
                ),
            )
        self.buttons = []
        columns = 4 if self.allow_background_capture else 3
        button_width = 150 if self.allow_background_capture else 200
        column_step = 158 if self.allow_background_capture else 210
        font_scale = 0.4 if self.allow_background_capture else .49
        for index, (key, label) in enumerate(labels):
            left = 10 + (index % columns) * column_step
            top = self.camera_height + 10 + (index // columns) * 52
            bounds = (left, top, left + button_width, top + 42)
            self.buttons.append((bounds, key, label))
            color = (62, 58, 48) if key != ord("q") else (55, 45, 135)
            cv2.rectangle(canvas, (left, top), (left + button_width, top + 42), color, -1)
            cv2.rectangle(canvas, (left, top), (left + button_width, top + 42), (140, 130, 110), 1)
            cv2.putText(canvas, label, (left + 10, top + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Click a button or use its keyboard shortcut.",
                    (12, self.camera_height + 122), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def show_camera(self, frame):
        # Mirror only the display, never the frame passed to the model.
        preview = cv2.flip(frame, 1)
        height, width = preview.shape[:2]
        preview = cv2.resize(preview, (640, max(1, round(height * 640 / width))))
        if self.effects is not None:
            self.effects.camera_height = preview.shape[0]
            for index, (x, y) in enumerate(self.effects.driver_anchors):
                point = (int((1 - x) * 640), int(y * preview.shape[0]))
                cv2.circle(preview, point, 7, (0, 255, 255), 2)
        if self.face_swap_mode:
            lines = [
                f"Camera {self.camera_index} | Mirrored preview | C: calibrate | Q: quit",
                self.status,
                f"Photo face + live camera body/background | "
                f"bounded head motion {'ON' if self.head_enabled else 'OFF'}",
            ]
        else:
            lines = [
                f"Camera {self.camera_index} | Mirrored input | C: calibrate | Q: quit",
                self.status,
                f"H: bounded head motion {'ON' if self.head_enabled else 'OFF'} | "
                f"L: shading {'ON' if self.effects and self.effects.shadows_enabled else 'OFF'} | B: shoulders off",
            ]
        if self.effects is not None:
            lines.append(self.effects.message)
        for index, line in enumerate(lines):
            position = (10, 24 + index * 27)
            cv2.putText(preview, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                        .47, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                        .47, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Your camera", self.add_controls(preview))

    def poll(self):
        key = cv2.waitKey(1) & 0xFF
        if key not in (ord("q"), 27) and self.pending_actions:
            key = self.pending_actions.popleft()
        self.last_key = key
        if key in (ord("q"), 27):
            return False
        if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
               for name in ("Your camera", "Render")):
            return False
        if key in (ord("c"), ord("C")):
            self.needs_reference = True
            self.calibration_requested = True
            self.status = "Calibrating on the next detected face..."
        if key in (ord("h"), ord("H")):
            self.head_enabled = not self.head_enabled
        if self.allow_background_capture and key in (ord("g"), ord("G")):
            self.background_capture_requested = True
            self.status = "Capturing room background. Stay out of frame."
        return True

    def capture_clean_background(
        self,
        capture,
        first_frame,
        target_shape,
        sample_count=7,
        sample_interval=0.12,
    ):
        """Capture a short median plate while the user is outside the camera view."""
        if sample_count < 1:
            raise ValueError("Background capture needs at least one frame.")

        frames = [first_frame]
        for sample_index in range(1, sample_count):
            if sample_interval > 0:
                time.sleep(sample_interval)
            ok, frame = capture.read(timeout=1.0)
            if not ok or frame is None:
                raise DriverBackgroundError(
                    "Could not read enough webcam frames for the clean background. "
                    "Check the camera, then press G to retry."
                )
            self.status = (
                f"Capturing clean room background {sample_index + 1}/{sample_count}. "
                "Stay out of frame."
            )
            self.show_camera(frame)
            if not self.poll():
                return None
            frames.append(frame)

        return median_clean_background(frames, target_shape)

    @property
    def ready(self):
        return not self.needs_reference or self.calibration_requested

    def tracking_result(self, detected):
        if not detected:
            self.status = (
                "No face detected; webcam body/background remain visible."
                if self.face_swap_mode
                else "No face detected. Portrait paused; look at the camera."
            )
            return
        self.needs_reference = False
        self.calibration_requested = False
        self.status = (
            "Photo face active; webcam body/background retained. Blink or speak to test expression."
            if self.face_swap_mode
            else "Face tracked. Blink or open your mouth to compare."
        )
