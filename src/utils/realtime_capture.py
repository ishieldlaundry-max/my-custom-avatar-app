"""Latest-frame webcam capture for low-latency realtime inference."""

import queue
import threading


class LatestFrameCapture:
    """Read webcam frames in the background and discard stale frames."""

    def __init__(self, capture):
        self.capture = capture
        self._frames = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._ended = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._worker,
            name="latest-frame-capture",
            daemon=True,
        )

    def start(self, timeout=3.0):
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError("Timed out while waiting for the first webcam frame.")
        if self._error is not None:
            raise RuntimeError(f"Webcam capture failed: {self._error}")
        return self

    def read(self, timeout=1.0):
        """Return the newest frame available, never a queued stale frame."""
        try:
            return True, self._frames.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise RuntimeError(f"Webcam capture failed: {self._error}")
            return False, None

    @property
    def ended(self):
        return self._ended.is_set()

    def close(self):
        self._stop.set()
        self.capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _worker(self):
        try:
            while not self._stop.is_set() and self.capture.isOpened():
                ok, frame = self.capture.read()
                if not ok:
                    break
                try:
                    self._frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    self._frames.put_nowait(frame)
                self._ready.set()
        except Exception as exc:
            self._error = exc
        finally:
            self._ended.set()
            self._ready.set()