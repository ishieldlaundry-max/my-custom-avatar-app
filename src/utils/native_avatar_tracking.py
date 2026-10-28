"""Local HTTP bridge from the embedded avatar preview to CUDA-first DWPose."""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np
from fastapi import HTTPException
from starlette.requests import Request

from .live_skeletal_tracking import LatestSkeletalInference


MAX_FRAME_BYTES = 1_500_000
MAX_FRAME_WIDTH = 1280
MAX_FRAME_HEIGHT = 960


class NativeAvatarTrackingSession:
    """Own one latest-frame DWPose worker for the local avatar preview."""

    def __init__(
        self,
        inference_factory: Callable[[], object] = LatestSkeletalInference,
    ):
        self._inference_factory = inference_factory
        self._lock = threading.RLock()
        self._inference = None

    def start(self) -> dict:
        with self._lock:
            if self._inference is None:
                self._inference = self._inference_factory().start()
            return self._snapshot_locked()

    def submit_jpeg(self, payload: bytes) -> dict:
        if not payload:
            raise ValueError("The webcam frame is empty.")
        if len(payload) > MAX_FRAME_BYTES:
            raise ValueError("The webcam frame exceeds the 1.5 MB limit.")
        with self._lock:
            inference = self._inference
            if inference is None:
                raise RuntimeError("Start native DWPose tracking before sending frames.")

        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("The webcam frame is not a valid JPEG image.")
        height, width = frame_bgr.shape[:2]
        if width > MAX_FRAME_WIDTH or height > MAX_FRAME_HEIGHT:
            raise ValueError(
                f"Webcam frames must be at most {MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}."
            )
        accepted = inference.submit(
            frame_bgr,
            time.monotonic_ns() // 1_000_000,
        )
        return {"accepted": bool(accepted)}

    def calibrate(self) -> dict:
        with self._lock:
            if self._inference is None:
                raise RuntimeError("Start native DWPose tracking before calibrating.")
            self._inference.request_calibration()
        return {"accepted": True}

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict:
        inference = self._inference
        if inference is None:
            return {
                "active": False,
                "status": "Native DWPose tracking is stopped.",
                "error": None,
                "pose": None,
                "inference_ms": 0.0,
                "dropped_frames": 0,
            }
        result = inference.snapshot()
        pose = result["frame"]
        return {
            "active": True,
            "status": result["status"],
            "error": result["error"],
            "pose": pose.to_dict() if pose is not None else None,
            "inference_ms": float(result["inference_ms"]),
            "dropped_frames": int(result["dropped_frames"]),
        }

    def stop(self) -> dict:
        with self._lock:
            inference = self._inference
            self._inference = None
        if inference is not None:
            inference.close()
        return self.snapshot()


def register_native_avatar_tracking_routes(app, session: NativeAvatarTrackingSession):
    """Attach loopback-friendly frame and control routes to the Gradio FastAPI app."""

    async def start_tracking():
        return session.start()

    async def send_frame(request: Request):
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type.lower() != "image/jpeg":
            raise HTTPException(status_code=415, detail="Expected an image/jpeg frame.")
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > MAX_FRAME_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="The webcam frame exceeds the 1.5 MB limit.",
                )
        try:
            return session.submit_jpeg(bytes(payload))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def request_calibration():
        try:
            return session.calibrate()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def get_snapshot():
        return session.snapshot()

    async def stop_tracking():
        return session.stop()

    app.add_api_route(
        "/native-avatar-tracking/start",
        start_tracking,
        methods=["POST"],
    )
    app.add_api_route(
        "/native-avatar-tracking/frame",
        send_frame,
        methods=["POST"],
    )
    app.add_api_route(
        "/native-avatar-tracking/calibrate",
        request_calibration,
        methods=["POST"],
    )
    app.add_api_route(
        "/native-avatar-tracking/snapshot",
        get_snapshot,
        methods=["GET"],
    )
    app.add_api_route(
        "/native-avatar-tracking/stop",
        stop_tracking,
        methods=["POST"],
    )
