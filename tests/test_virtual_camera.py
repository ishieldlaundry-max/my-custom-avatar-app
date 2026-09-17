import time

import numpy as np

from src.utils.virtual_camera import VirtualCameraBroadcaster


class FakeCamera:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
        self.closed = False

    def send(self, frame):
        self.frames.append(frame.copy())

    def sleep_until_next_frame(self):
        time.sleep(1.0 / self.fps)

    def close(self):
        self.closed = True


def test_broadcaster_resizes_and_closes_without_blocking():
    cameras = []

    def factory(**kwargs):
        camera = FakeCamera(**kwargs)
        cameras.append(camera)
        return camera

    broadcaster = VirtualCameraBroadcaster(64, 48, fps=30, camera_factory=factory).start()
    broadcaster.send(np.full((24, 32, 3), 127, dtype=np.uint8))

    deadline = time.time() + 1.0
    while not cameras[0].frames and time.time() < deadline:
        time.sleep(0.01)
    broadcaster.close()

    assert cameras[0].frames[0].shape == (48, 64, 3)
    assert cameras[0].frames[0].dtype == np.uint8
    np.testing.assert_array_equal(cameras[0].frames[0], 127)
    assert cameras[0].closed


def test_broadcaster_reports_backend_startup_failure():
    def factory(**kwargs):
        raise RuntimeError("no virtual camera backend")

    try:
        VirtualCameraBroadcaster(64, 48, camera_factory=factory).start()
    except RuntimeError as exc:
        assert "no virtual camera backend" in str(exc)
    else:
        raise AssertionError("Expected virtual camera startup to fail.")


def test_broadcaster_repeats_latest_frame_at_fixed_rate():
    cameras = []

    def factory(**kwargs):
        camera = FakeCamera(**kwargs)
        cameras.append(camera)
        return camera

    broadcaster = VirtualCameraBroadcaster(16, 16, fps=30, camera_factory=factory).start()
    broadcaster.send(np.zeros((16, 16, 3), dtype=np.uint8))
    time.sleep(0.09)
    broadcaster.close()

    assert len(cameras[0].frames) >= 2