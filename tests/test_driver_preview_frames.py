import cv2
import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("height", "width"),
    [(1200, 800), (800, 1200)],
    ids=["portrait", "landscape"],
)
def test_extract_driver_preview_frames_preserves_all_margins(
    monkeypatch, tmp_path, height, width
):
    video = tmp_path / "framed.mp4"
    video.write_bytes(b"placeholder")
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    markers = [
        ((10, 20, 30), (30, 20, 10)),
        ((40, 50, 60), (60, 50, 40)),
        ((70, 80, 90), (90, 80, 70)),
        ((100, 110, 120), (120, 110, 100)),
    ]
    frame[:40, :40] = markers[0][0]
    frame[:40, -40:] = markers[1][0]
    frame[-40:, :40] = markers[2][0]
    frame[-40:, -40:] = markers[3][0]

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1
            if prop == cv2.CAP_PROP_FPS:
                return 30
            return 0

        def set(self, _prop, _value):
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture())
    previews = extract_driver_preview_frames(video, max_frames=1, max_edge=600)

    preview = previews[0][0]
    scale = min(1.0, 600 / max(height, width))
    assert preview.shape == (
        round(height * scale),
        round(width * scale),
        3,
    )
    for point, (_, expected_rgb) in zip(
        [(4, 4), (4, -5), (-5, 4), (-5, -5)],
        markers,
        strict=True,
    ):
        assert tuple(preview[point[0], point[1]]) == expected_rgb
