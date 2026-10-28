"""CUDA-only DWPose backend and bounded asynchronous inference worker."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from .skeletal_frame import BODY_CONNECTIONS, SkeletalFrame
from .skeletal_pose_mapper import SkeletalPoseMapper


class DWPoseCudaTracker:
    """Run the installed DWPose Wholebody ONNX models with CUDA only."""

    def __init__(self, backend_root: Optional[str | Path] = None):
        self.backend_root = Path(
            backend_root
            or Path(__file__).resolve().parents[2] / "third_party" / "MimicMotion"
        ).expanduser().resolve()
        model_dir = self.backend_root / "models" / "DWPose"
        self.detector_path = model_dir / "yolox_l.onnx"
        self.pose_path = model_dir / "dw-ll_ucoco_384.onnx"
        missing = [
            path.name for path in (self.detector_path, self.pose_path)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "DWPose model assets are missing: "
                + ", ".join(missing)
                + ". Run setup_full_body_windows.bat before starting native skeletal tracking."
            )

        if str(self.backend_root) not in sys.path:
            sys.path.insert(0, str(self.backend_root))
        try:
            ort = importlib.import_module("onnxruntime")
            detector_module = importlib.import_module("mimicmotion.dwpose.onnxdet")
            pose_module = importlib.import_module("mimicmotion.dwpose.onnxpose")
        except Exception as exc:
            raise RuntimeError(
                "Could not load the DWPose ONNX runtime helpers from "
                f"{self.backend_root}. Run setup_full_body_windows.bat and check "
                "the main app's Python environment."
            ) from exc

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "Native skeletal tracking requires ONNX Runtime CUDA in the main "
                "application environment."
            )
        if "CPUExecutionProvider" not in available_providers:
            raise RuntimeError(
                "Native skeletal tracking requires ONNX Runtime CPU placement "
                "for two profiled pose operators."
            )
        strict_cuda_options = ort.SessionOptions()
        strict_cuda_options.intra_op_num_threads = 1
        strict_cuda_options.inter_op_num_threads = 1
        strict_cuda_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        strict_cuda_options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", "1"
        )
        pose_options = ort.SessionOptions()
        pose_options.intra_op_num_threads = 1
        pose_options.inter_op_num_threads = 1
        cv2.setNumThreads(2)
        cuda_provider = ("CUDAExecutionProvider", {"device_id": 0})
        self.detector_session = ort.InferenceSession(
            str(self.detector_path),
            sess_options=strict_cuda_options,
            providers=[cuda_provider],
        )
        self.pose_session = ort.InferenceSession(
            str(self.pose_path),
            sess_options=pose_options,
            providers=[cuda_provider, "CPUExecutionProvider"],
        )
        for session in (self.detector_session, self.pose_session):
            if "CUDAExecutionProvider" not in session.get_providers():
                self.close()
                raise RuntimeError(
                    "DWPose did not initialize with CUDAExecutionProvider. "
                    "CPU fallback is disabled."
                )
            session.disable_fallback()
        self._inference_detector = detector_module.inference_detector
        self._inference_pose = pose_module.inference_pose

    def track(self, frame_bgr: np.ndarray) -> dict:
        height, width = frame_bgr.shape[:2]
        boxes = self._inference_detector(self.detector_session, frame_bgr)
        keypoints, scores = self._inference_pose(
            self.pose_session, boxes, frame_bgr
        )
        info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        if info.shape[0] == 0:
            return {"bodies": {"subset": [], "candidate": [], "score": []}}

        # DWPose's MMPose layout has no neck joint; insert one and reorder its
        # body points to the OpenPose 18-point convention used by MimicMotion.
        neck = np.mean(info[:, [5, 6]], axis=1)
        neck[:, 2] = np.logical_and(info[:, 5, 2] > 0.3, info[:, 6, 2] > 0.3)
        info = np.insert(info, 17, neck, axis=1)
        mmpose_indices = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        info[:, openpose_indices] = info[:, mmpose_indices]
        info = info.copy()
        info[:, :, 0] /= float(width)
        info[:, :, 1] /= float(height)
        confidence = np.clip(np.nan_to_num(info[:, :, 2]), 0.0, 1.0)
        people = len(info)
        body = info[:, :18, :].reshape((people * 18, 3)).copy()
        subset = confidence[:, :18].copy()
        for person_index in range(people):
            for joint_index in range(18):
                subset[person_index, joint_index] = (
                    person_index * 18 + joint_index
                    if subset[person_index, joint_index] > 0.3 else -1
                )
        hands = np.concatenate(
            (info[:, 92:113, :], info[:, 113:134, :]), axis=0
        )
        return {
            "bodies": {
                "candidate": body,
                "subset": subset,
                "score": confidence[:, :18].copy(),
            },
            "faces": info[:, 24:92, :].copy(),
            "faces_score": confidence[:, 24:92].copy(),
            "hands": hands.copy(),
            "hands_score": np.concatenate(
                (confidence[:, 92:113], confidence[:, 113:134]), axis=0
            ),
        }

    def close(self) -> None:
        self.detector_session = None
        self.pose_session = None
        gc.collect()


class LatestSkeletalInference:
    """Run inference asynchronously while keeping only the newest queued frame."""

    def __init__(
        self,
        tracker_factory: Callable[[], object] = DWPoseCudaTracker,
        mapper_factory: Callable[[], SkeletalPoseMapper] = SkeletalPoseMapper,
    ):
        self.tracker_factory = tracker_factory
        self.mapper_factory = mapper_factory
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._calibrate = threading.Event()
        self._reset_mapper = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker,
            name="skeletal-onnx-inference",
            daemon=True,
        )
        self._latest: Optional[SkeletalFrame] = None
        self._status = "Loading CUDA DWPose models."
        self._error: Optional[str] = None
        self._dropped_frames = 0
        self._inference_ms = 0.0
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("Skeletal inference worker already started.")
        self._started = True
        self._thread.start()
        return self

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int) -> bool:
        if self._stop.is_set() or self._error:
            return False
        packet = (np.asarray(frame_bgr).copy(), int(timestamp_ms))
        try:
            self._frames.put_nowait(packet)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped_frames += 1
            self._frames.put_nowait(packet)
        return True

    def request_calibration(self) -> None:
        self._calibrate.set()

    def request_mapper_reset(self) -> None:
        """Discard camera-specific calibration while keeping the models loaded."""
        self._reset_mapper.set()
        with self._lock:
            self._latest = None
            self._status = "Camera changed. Press C in a neutral full-body pose."

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frame": self._latest,
                "status": self._status,
                "error": self._error,
                "dropped_frames": self._dropped_frames,
                "inference_ms": self._inference_ms,
            }

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                pass
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _worker(self) -> None:
        tracker = None
        try:
            tracker = self.tracker_factory()
            mapper = self.mapper_factory()
            self._set_status("CUDA DWPose ready. Press C in a neutral full-body pose.")
            while not self._stop.is_set():
                try:
                    packet = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if packet is None:
                    break
                if self._reset_mapper.is_set():
                    self._reset_mapper.clear()
                    mapper = self.mapper_factory()
                    self._calibrate.clear()
                    self._set_status(
                        "Camera changed. Press C in a neutral full-body pose."
                    )
                frame, timestamp_ms = packet
                started = time.perf_counter()
                pose = tracker.track(frame)
                if self._calibrate.is_set():
                    self._calibrate.clear()
                    try:
                        mapper.calibrate(pose)
                        self._set_status("Full-body calibration saved.")
                    except ValueError as exc:
                        self._set_status(f"Calibration failed: {exc}")
                skeletal = mapper.update(pose, timestamp_ms, frame.shape)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._latest = skeletal
                    self._inference_ms = elapsed_ms
                    if skeletal.tracked and not skeletal.calibrated:
                        self._status = (
                            "Tracking. Press C in a neutral full-body pose to calibrate."
                        )
                    elif not skeletal.tracked:
                        self._status = "No person detected. Keep your full body in view."
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._status = "Skeletal tracking stopped because inference failed."
        finally:
            if tracker is not None:
                try:
                    tracker.close()
                except Exception as exc:
                    with self._lock:
                        self._error = self._error or (
                            f"Tracker shutdown failed: {type(exc).__name__}: {exc}"
                        )
            self._set_status(
                self._status if self._error else "Skeletal tracking stopped."
            )
