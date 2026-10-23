"""Optional 2-D shoulder deformation and illustrative directional shading.

These are image-processing effects, not inferred 3-D anatomy or physical light.
"""
import cv2
import numpy as np

from .live_motion import bounded_pose


class ShoulderFlow:
    """Track clothing features near user-marked shoulders with FB validation."""
    def __init__(self):
        self.previous = None
        self.groups = []
        self.offsets = np.zeros((2, 2), np.float32)
        self.valid = False

    def reset(self, frame, anchors):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous = gray
        self.groups = []
        self.offsets[:] = 0
        h, w = gray.shape
        for anchor in anchors:
            x, y = np.asarray(anchor) * (w, h)
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), max(12, int(w * .06)), 255, -1)
            points = cv2.goodFeaturesToTrack(gray, 35, .015, 4, mask=mask)
            self.groups.append(points)
        self.valid = len(self.groups) == 2 and all(
            p is not None and len(p) >= 4 for p in self.groups
        )
        return self.valid

    def update(self, frame):
        if not self.valid:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        next_groups, steps = [], []
        for points in self.groups:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous, gray, points, None, winSize=(25, 25), maxLevel=3)
            if moved is None:
                self.valid = False
                return None
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.previous, moved, None, winSize=(25, 25), maxLevel=3)
            if back is None:
                self.valid = False
                return None
            keep = ((status.ravel() == 1) & (back_status.ravel() == 1)
                    & (np.linalg.norm(back - points, axis=2).ravel() < 1.5))
            if keep.sum() < 4:
                self.valid = False
                return None
            step = np.median((moved - points)[keep, 0], axis=0)
            if np.linalg.norm(step / (w, h)) > .06:
                self.valid = False
                return None
            steps.append(step / (w, h))
            next_groups.append(moved[keep])
        candidate = self.offsets + np.asarray(steps, np.float32)
        # Do not keep accumulating a clipped motion that would take ages to return.
        if np.max(np.abs(candidate)) > .12:
            self.valid = False
            return None
        self.offsets = candidate
        self.previous, self.groups = gray, next_groups
        return self.offsets.copy()


def deform_shoulders(image, anchors, offsets):
    """Local inverse warp, fading above the shoulders to protect the face."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy, total = np.zeros_like(xx), np.zeros_like(xx), np.zeros_like(xx)
    for (ax, ay), (ox, oy) in zip(anchors, offsets):
        weight = np.exp(-.5 * (((xx / w - ax) / .18) ** 2
                              + ((yy / h - ay) / .22) ** 2))
        weight *= np.clip((yy / h - (ay - .15)) / .15, 0, 1)
        dx += weight * np.clip(ox, -.045, .045) * w
        dy += weight * np.clip(oy, -.045, .045) * h
        total += weight
    normalizer = np.maximum(total, 1)
    return cv2.remap(image, xx - dx / normalizer, yy - dy / normalizer,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def shade_portrait(image, face_bounds, pose):
    """Low-amplitude face-local shading; zero at the calibrated neutral pose."""
    pitch, yaw, _ = (float(np.asarray(p).ravel()[0]) for p in pose)
    x0, y0, x1, y1 = face_bounds
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    nx = (xx - (x0 + x1) / 2) / max((x1 - x0) * .7, 1)
    ny = (yy - (y0 + y1) / 2) / max((y1 - y0) * .8, 1)
    mask = np.clip(1 - nx ** 2 - ny ** 2, 0, 1) ** 2
    gain = 1 + mask * np.clip(-.06 * yaw / 15 * nx + .04 * pitch / 10 * ny, -.08, .08)
    return np.clip(image.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


class LivePortraitEffects:
    def __init__(self, target_rgb, face_landmarks):
        self.target_shape = target_rgb.shape
        points = np.asarray(face_landmarks).reshape(-1, 2)
        self.face_bounds = (*points.min(axis=0), *points.max(axis=0))
        self.driver_anchors, self.target_anchors = [], []
        self.selecting = False
        self.shoulders_enabled = False
        self.shadows_enabled = True
        self.flow = ShoulderFlow()
        self.reference = None
        self.camera_height = 480
        self.message = "S: mark shoulders in both windows (2D effect)."
        # Camera clicks are routed by LiveTrackingPreview so its toolbar wins.
        cv2.setMouseCallback("Render", self._target_click)

    def _driver_click(self, event, x, y, flags, param):
        if self.selecting and event == cv2.EVENT_LBUTTONDOWN and len(self.driver_anchors) < 2:
            # Your camera is mirrored and resized to 640 px.
            self.driver_anchors.append((1 - x / 640, y / self.camera_height))
            self._selection_status()

    def _target_click(self, event, x, y, flags, param):
        if self.selecting and event == cv2.EVENT_LBUTTONDOWN and len(self.target_anchors) < 2:
            h, w = self.target_shape[:2]
            self.target_anchors.append((x / w, y / h))
            self._selection_status()

    def _selection_status(self):
        self.message = (f"Select shoulders: camera {len(self.driver_anchors)}/2, "
                        f"portrait {len(self.target_anchors)}/2. Then C.")
        if len(self.driver_anchors) == len(self.target_anchors) == 2:
            if any(np.linalg.norm(np.asarray(points[0]) - points[1]) < .12
                   for points in (self.driver_anchors, self.target_anchors)):
                self.message = "Shoulder points too close together. Press S to select again."
                return
            self.driver_anchors.sort()
            self.target_anchors.sort()
            self.selecting = False
            self.shoulders_enabled = True
            self.message = "Shoulders selected. Hold still and press C."

    def handle_key(self, key):
        if key in (ord("s"), ord("S")):
            self.selecting = True
            self.shoulders_enabled = False
            self.driver_anchors, self.target_anchors = [], []
            self._selection_status()
        elif key in (ord("b"), ord("B")):
            self.shoulders_enabled = False
            self.selecting = False
            self.message = "Shoulder effect off. S: set again."
        elif key in (ord("l"), ord("L")):
            self.shadows_enabled = not self.shadows_enabled

    def selection_image(self, target_rgb):
        image = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        for x, y in self.target_anchors:
            cv2.circle(image, (int(x * w), int(y * h)), 9, (0, 255, 255), 3)
        return image

    def apply(self, frame, output_rgb, info, reset=False, head_enabled=True):
        if reset or self.reference is None:
            self.reference = {k: np.asarray(info[k]).copy() for k in ("pitch", "yaw", "roll")}
            if self.shoulders_enabled:
                ok = self.flow.reset(frame, self.driver_anchors)
                self.message = "Shoulders tracking (2D)." if ok else "No shoulder detail; S: select textured clothing."
        result = output_rgb
        if self.shoulders_enabled and self.flow.groups:
            offsets = self.flow.update(frame) if self.flow.valid else None
            if offsets is None:
                self.message = "Shoulders lost; hold still and press C to reset."
                offsets = self.flow.offsets
            result = deform_shoulders(result, self.target_anchors, offsets)
        if self.shadows_enabled:
            pose = bounded_pose(info, self.reference) if head_enabled else (0, 0, 0)
            result = shade_portrait(result, self.face_bounds, pose)
        return result
