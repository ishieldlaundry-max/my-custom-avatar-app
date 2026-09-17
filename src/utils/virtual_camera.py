"""Non-blocking virtual-camera output for local portrait rendering."""

import queue
import threading

import cv2
import numpy as np


class VirtualCameraBroadcaster:
    """Publish the newest RGB frame without blocking the inference loop."""

    def __init__(self, width, height, fps=30, camera_factory=None):
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(round(fps or 30)))
        self._camera_factory = camera_factory
        self._frames = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._worker,
            name="virtual-camera-output",
            daemon=True,
        )

    def start(self, timeout=5.0):
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError("Timed out while starting the virtual camera.")
        if self._error is not None:
            raise RuntimeError(f"Could not start virtual camera: {self._error}")
        return self

    def send(self, frame):
        """Queue an RGB uint8 frame, replacing a stale frame when necessary."""
        if self._error is not None:
            raise RuntimeError(f"Virtual camera stopped: {self._error}")
        if self._stop.is_set():
            return
        prepared = self._prepare_frame(frame)
        try:
            self._frames.put_nowait(prepared)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(prepared)

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _prepare_frame(self, frame):
        prepared = np.asarray(frame)
        if prepared.ndim != 3 or prepared.shape[2] != 3:
            raise ValueError("Virtual camera frames must be HWC RGB images.")
        if prepared.shape[:2] != (self.height, self.width):
            prepared = cv2.resize(
                prepared,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(np.clip(prepared, 0, 255).astype(np.uint8))

    def _worker(self):
        camera = None
        try:
            if self._camera_factory is None:
                try:
                    import pyvirtualcam
                except ImportError as exc:
                    raise RuntimeError(
                        "pyvirtualcam is not installed. Install requirements_win.txt "
                        "in the local TensorRT environment."
                    ) from exc
                camera = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.RGB,
                )
            else:
                camera = self._camera_factory(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                )
            self._ready.set()
            latest_frame = None
            while not self._stop.is_set():
                try:
                    latest_frame = self._frames.get(
                        timeout=0.1 if latest_frame is None else 0
                    )
                except queue.Empty:
                    if latest_frame is None:
                        continue
                camera.send(latest_frame)
                camera.sleep_until_next_frame()
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            if camera is not None:
                camera.close()
            self._ready.set()