import time

import numpy as np

from src.utils.realtime_capture import LatestFrameCapture


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return not self.released and bool(self.frames)

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def test_latest_frame_capture_drops_stale_frames_and_closes():
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(5)]
    capture = FakeCapture(frames)
    reader = LatestFrameCapture(capture).start()
    time.sleep(0.02)

    ok, frame = reader.read()
    reader.close()

    assert ok
    assert np.all(frame == 4)
    assert capture.released