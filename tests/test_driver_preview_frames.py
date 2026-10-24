import cv2
import numpy as np

from src.pipelines.full_body_animation_pipeline import extract_driver_preview_frames


def test_extract_driver_preview_frames_samples_evenly_spaced_moments(monkeypatch, tmp_path):
    video = tmp_path / "moments.mp4"
    video.write_bytes(b"placeholder")
    captures = []

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0
            self.positions_read = []
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10
            if prop == cv2.CAP_PROP_FPS:
                return 10
            return 0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)
            return True

        def read(self):
            self.positions_read.append(self.position)
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            frame[:, :, 0] = self.position
            return True, frame

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    previews = extract_driver_preview_frames(video)

    assert len(previews) == 4
    assert captures[0].positions_read == [0, 3, 6, 9]
    assert captures[0].released
    assert [label for _, label in previews] == ["0.0 s", "0.3 s", "0.6 s", "0.9 s"]
    assert [tuple(frame[0, 0]) for frame, _ in previews] == [
        (0, 0, 0),
        (0, 0, 3),
        (0, 0, 6),
        (0, 0, 9),
    ]
    assert all(frame.shape == (40, 60, 3) for frame, _ in previews)


def test_extract_driver_preview_frames_returns_empty_for_missing_video(tmp_path):
    assert extract_driver_preview_frames(tmp_path / "missing.mp4") == []
