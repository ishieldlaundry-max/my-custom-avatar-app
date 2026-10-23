"""OpenCV diagnostics for the exact camera frames supplied to portrait inference."""
from pathlib import Path

import cv2


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
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.needs_reference = True
        self.calibration_requested = False
        self.head_enabled = True
        self.last_key = -1
        self.effects = None
        self.status = "Hold a neutral face, eyes open. Press C to calibrate."
        for title, x in (("Your camera", 20), ("Render", 690)):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640, 560)
            cv2.moveWindow(title, x, 60)

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
        cv2.imshow("Your camera", preview)

    def poll(self):
        key = cv2.waitKey(1) & 0xFF
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
        return True

    @property
    def ready(self):
        return not self.needs_reference or self.calibration_requested

    def tracking_result(self, detected):
        if not detected:
            self.status = "No face detected. Portrait paused; look at the camera."
            return
        self.needs_reference = False
        self.calibration_requested = False
        self.status = "Face tracked. Blink or open your mouth to compare."
