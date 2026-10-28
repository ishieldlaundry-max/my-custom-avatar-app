import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src.utils.native_avatar_tracking import (
    MAX_FRAME_BYTES,
    NativeAvatarTrackingSession,
    register_native_avatar_tracking_routes,
)


class FakeFrame:
    def to_dict(self):
        return {"tracked": True, "timestamp_ms": 123}


class FakeInference:
    def __init__(self):
        self.submitted = []
        self.calibration_requested = False
        self.closed = False

    def start(self):
        return self

    def submit(self, frame, timestamp_ms):
        self.submitted.append((frame, timestamp_ms))
        return True

    def request_calibration(self):
        self.calibration_requested = True

    def snapshot(self):
        return {
            "frame": FakeFrame(),
            "status": "CUDA DWPose ready.",
            "error": None,
            "inference_ms": 12.5,
            "dropped_frames": 2,
        }

    def close(self):
        self.closed = True


def jpeg_frame(width=80, height=60):
    ok, encoded = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    assert ok
    return encoded.tobytes()


def test_native_avatar_tracking_session_relays_frames_and_lifecycle():
    worker = FakeInference()
    session = NativeAvatarTrackingSession(lambda: worker)

    started = session.start()
    assert started["active"] is True
    assert started["pose"] == {"tracked": True, "timestamp_ms": 123}
    assert session.submit_jpeg(jpeg_frame()) == {"accepted": True}
    assert worker.submitted[0][0].shape == (60, 80, 3)
    assert isinstance(worker.submitted[0][1], int)

    assert session.calibrate() == {"accepted": True}
    assert worker.calibration_requested is True
    assert session.stop()["active"] is False
    assert worker.closed is True


def test_native_avatar_tracking_session_rejects_invalid_frames():
    session = NativeAvatarTrackingSession(lambda: FakeInference())
    with pytest.raises(RuntimeError, match="Start native DWPose"):
        session.submit_jpeg(jpeg_frame())

    session.start()
    with pytest.raises(ValueError, match="valid JPEG"):
        session.submit_jpeg(b"not a JPEG")
    with pytest.raises(ValueError, match="1.5 MB"):
        session.submit_jpeg(b"x" * (MAX_FRAME_BYTES + 1))
    with pytest.raises(ValueError, match="at most"):
        session.submit_jpeg(jpeg_frame(width=1281, height=64))


def test_native_avatar_tracking_routes_handle_frame_control_and_status():
    worker = FakeInference()
    session = NativeAvatarTrackingSession(lambda: worker)
    app = FastAPI()
    register_native_avatar_tracking_routes(app, session)
    client = TestClient(app)

    assert client.get("/native-avatar-tracking/snapshot").json()["active"] is False
    assert client.post(
        "/native-avatar-tracking/frame",
        content=jpeg_frame(),
        headers={"content-type": "image/jpeg"},
    ).status_code == 409
    assert client.post("/native-avatar-tracking/start").status_code == 200
    assert client.post(
        "/native-avatar-tracking/frame",
        content=jpeg_frame(),
        headers={"content-type": "image/jpeg"},
    ).json() == {"accepted": True}
    assert client.post("/native-avatar-tracking/calibrate").json() == {"accepted": True}
    snapshot = client.get("/native-avatar-tracking/snapshot").json()
    assert snapshot["pose"]["tracked"] is True
    assert snapshot["inference_ms"] == 12.5
    assert client.post("/native-avatar-tracking/stop").json()["active"] is False
    assert worker.closed is True
