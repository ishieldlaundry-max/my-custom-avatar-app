import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.live_skeletal_tracking import (
    DWPoseCudaTracker,
    LatestSkeletalInference,
    SkeletalPoseMapper,
)
from src.utils.live_skeletal_preview import render_camera_view, render_skeleton_view


BODY_POINTS = np.array(
    [
        [0.50, 0.10], [0.50, 0.18], [0.42, 0.25], [0.32, 0.28],
        [0.22, 0.25], [0.58, 0.25], [0.68, 0.28], [0.78, 0.25],
        [0.45, 0.50], [0.45, 0.70], [0.45, 0.92], [0.55, 0.50],
        [0.55, 0.70], [0.55, 0.92], [0.48, 0.12], [0.52, 0.12],
        [0.44, 0.15], [0.56, 0.15],
    ],
    dtype=np.float32,
)


def make_pose(x_shift=0.0, scale_y=1.0, confidence=0.9):
    body = np.zeros((18, 3), dtype=np.float32)
    body[:, :2] = BODY_POINTS
    body[:, 0] += x_shift
    body[:, 1] = 0.5 + (body[:, 1] - 0.5) * scale_y
    body[:, 2] = confidence

    face = np.zeros((1, 68, 3), dtype=np.float32)
    face[:, :, 0] = 0.5 + x_shift
    face[:, :, 1] = 0.13
    face[:, :, 2] = confidence
    face_points = {
        36: (0.47, 0.13), 37: (0.475, 0.125), 38: (0.485, 0.125),
        39: (0.49, 0.13), 40: (0.485, 0.135), 41: (0.475, 0.135),
        42: (0.51, 0.13), 43: (0.515, 0.125), 44: (0.525, 0.125),
        45: (0.53, 0.13), 46: (0.525, 0.135), 47: (0.515, 0.135),
        48: (0.475, 0.16), 51: (0.49, 0.155), 54: (0.525, 0.16),
        57: (0.50, 0.17), 62: (0.49, 0.16), 66: (0.49, 0.162),
    }
    for index, (x, y) in face_points.items():
        face[0, index] = (x + x_shift, y, confidence)

    hands = np.zeros((2, 21, 3), dtype=np.float32)
    for hand_index, wrist_x in enumerate((body[4, 0], body[7, 0])):
        for joint_index in range(21):
            hands[hand_index, joint_index] = (
                wrist_x + (joint_index % 4) * 0.004,
                body[4 if hand_index == 0 else 7, 1] + (joint_index // 4) * 0.004,
                confidence,
            )
    return {
        "bodies": {
            "candidate": body.copy(),
            "subset": np.arange(18, dtype=np.float32)[None, :],
            "score": np.full((1, 18), confidence, dtype=np.float32),
        },
        "faces": face,
        "faces_score": np.full((1, 68), confidence, dtype=np.float32),
        "hands": hands,
        "hands_score": np.full((2, 21), confidence, dtype=np.float32),
    }


def wait_for_frame(worker, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = worker.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"Condition not reached: {worker.snapshot()}")


def test_mapper_calibrates_and_returns_serializable_2_5d_frame():
    mapper = SkeletalPoseMapper()
    pose = make_pose()
    mapper.calibrate(pose)

    frame = mapper.update(pose, 100, (480, 640, 3))

    assert frame.tracked and frame.calibrated
    assert frame.root_translation == pytest.approx((0.0, 0.0, 0.0), abs=0.01)
    assert frame.depth_mode == "monocular_body_scale_estimate"
    assert "left_wrist" in frame.landmarks
    assert "LeftIndexProximal" in frame.bones
    assert set(frame.blendshapes) == {
        "eyeBlinkLeft", "eyeBlinkRight", "jawOpen",
        "mouthSmileLeft", "mouthSmileRight",
    }
    json.dumps(frame.to_dict())


def test_mapper_tracks_relative_translation_and_scale_depth():
    mapper = SkeletalPoseMapper()
    mapper.calibrate(make_pose())
    baseline = mapper.update(make_pose(), 67, (480, 640, 3))

    moved = mapper.update(make_pose(x_shift=0.10, scale_y=0.90), 100, (480, 640, 3))

    assert moved.root_translation[0] > 0.04
    assert moved.root_translation[2] > 0.02
    assert moved.root_translation[2] < 0.2
    assert moved.landmarks["nose"][0] == pytest.approx(
        baseline.landmarks["nose"][0], abs=0.01
    )
    assert moved.landmarks["nose"][2] == pytest.approx(0.0)


def test_mapper_handles_tracking_loss_with_confidence_aware_recovery():
    mapper = SkeletalPoseMapper()
    mapper.calibrate(make_pose())
    tracked = mapper.update(make_pose(x_shift=0.10), 100, (480, 640, 3))

    lost = mapper.update({"bodies": {"subset": [], "candidate": []}}, 133, (480, 640, 3))

    assert tracked.tracked
    assert not lost.tracked
    assert lost.calibrated
    assert abs(lost.root_translation[0]) < abs(tracked.root_translation[0])
    assert all(np.isfinite(value) for point in lost.landmarks.values() for value in point)
    assert lost.foot_contacts == {"left": False, "right": False}


def test_mapper_refuses_calibration_without_a_person():
    mapper = SkeletalPoseMapper()
    with pytest.raises(ValueError, match="No reliable body pose"):
        mapper.calibrate({"bodies": {"subset": [], "candidate": []}})


def test_latest_inference_worker_drops_stale_frames_and_calibrates():
    class FakeTracker:
        def track(self, frame):
            time.sleep(0.025)
            return make_pose(x_shift=float(frame[0, 0, 0]) / 2550.0)

        def close(self):
            pass

    worker = LatestSkeletalInference(tracker_factory=FakeTracker).start()
    try:
        for value in range(10):
            worker.submit(np.full((8, 8, 3), value * 20, dtype=np.uint8), value * 33)
        first = wait_for_frame(worker, lambda value: value["frame"] is not None)
        assert first["dropped_frames"] > 0

        worker.request_calibration()
        worker.submit(np.zeros((8, 8, 3), dtype=np.uint8), 400)
        calibrated = wait_for_frame(
            worker,
            lambda value: value["frame"] is not None and value["frame"].calibrated,
        )
        assert calibrated["frame"].tracked
    finally:
        worker.close()


def test_latest_inference_worker_surfaces_backend_errors():
    def fail():
        raise RuntimeError("CUDA provider unavailable")

    worker = LatestSkeletalInference(tracker_factory=fail).start()
    try:
        snapshot = wait_for_frame(worker, lambda value: value["error"] is not None)
        assert "CUDA provider unavailable" in snapshot["error"]
    finally:
        worker.close()


def test_dwp_pose_tracker_requires_provisioned_model_assets(tmp_path):
    with pytest.raises(RuntimeError, match="DWPose model assets are missing"):
        DWPoseCudaTracker(tmp_path)


def test_dwp_pose_tracker_refuses_cpu_only_onnx_runtime(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "DWPose"
    model_dir.mkdir(parents=True)
    (model_dir / "yolox_l.onnx").write_bytes(b"detector")
    (model_dir / "dw-ll_ucoco_384.onnx").write_bytes(b"pose")

    modules = {
        "onnxruntime": SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        ),
        "mimicmotion.dwpose.onnxdet": SimpleNamespace(),
        "mimicmotion.dwpose.onnxpose": SimpleNamespace(),
    }
    monkeypatch.setattr(
        "src.utils.live_skeletal_tracking.importlib.import_module",
        lambda name: modules[name],
    )

    with pytest.raises(RuntimeError, match="requires ONNX Runtime CUDA"):
        DWPoseCudaTracker(tmp_path)


def test_dwp_pose_tracker_allows_cpu_only_for_pose_graph(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "DWPose"
    model_dir.mkdir(parents=True)
    detector_path = model_dir / "yolox_l.onnx"
    pose_path = model_dir / "dw-ll_ucoco_384.onnx"
    detector_path.write_bytes(b"detector")
    pose_path.write_bytes(b"pose")

    class Options:
        def __init__(self):
            self.entries = {}
            self.graph_optimization_level = None

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    sessions = []
    class Session:
        def __init__(self, model, sess_options, providers):
            self.model = model
            self.options = sess_options
            self.providers = providers
            self.disabled_runtime_fallback = False
            sessions.append(self)

        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def disable_fallback(self):
            self.disabled_runtime_fallback = True

    modules = {
        "onnxruntime": SimpleNamespace(
            SessionOptions=Options,
            GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL="disable_all"),
            get_available_providers=lambda: [
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ],
            InferenceSession=Session,
        ),
        "mimicmotion.dwpose.onnxdet": SimpleNamespace(inference_detector=lambda *_: []),
        "mimicmotion.dwpose.onnxpose": SimpleNamespace(inference_pose=lambda *_: []),
    }
    monkeypatch.setattr("src.utils.live_skeletal_tracking.cv2.setNumThreads", lambda _: None)
    monkeypatch.setattr(
        "src.utils.live_skeletal_tracking.importlib.import_module",
        lambda name: modules[name],
    )

    tracker = DWPoseCudaTracker(tmp_path)

    assert [session.model for session in sessions] == [
        str(detector_path), str(pose_path)
    ]
    assert sessions[0].options.entries["session.disable_cpu_ep_fallback"] == "1"
    assert sessions[0].options.graph_optimization_level == "disable_all"
    assert "session.disable_cpu_ep_fallback" not in sessions[1].options.entries
    assert sessions[1].options.graph_optimization_level is None
    assert sessions[0].providers == [("CUDAExecutionProvider", {"device_id": 0})]
    assert sessions[1].providers == [
        ("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"
    ]
    assert all(session.disabled_runtime_fallback for session in sessions)
    tracker.close()

def test_dwp_pose_adapter_returns_body_hands_and_face_keypoints():
    tracker = DWPoseCudaTracker.__new__(DWPoseCudaTracker)
    tracker.detector_session = object()
    tracker.pose_session = object()
    tracker._inference_detector = lambda session, image: "boxes"
    keypoints = np.zeros((1, 133, 2), dtype=np.float32)
    keypoints[0, :, 0] = np.arange(133, dtype=np.float32) + 10
    keypoints[0, :, 1] = np.arange(133, dtype=np.float32) + 20
    scores = np.full((1, 133), 0.9, dtype=np.float32)
    tracker._inference_pose = lambda session, boxes, image: (keypoints, scores)

    pose = tracker.track(np.zeros((200, 200, 3), dtype=np.uint8))

    assert pose["bodies"]["candidate"].shape == (18, 3)
    assert pose["faces"].shape == (1, 68, 3)
    assert pose["hands"].shape == (2, 21, 3)
    assert np.all(pose["bodies"]["subset"] >= 0)
    assert np.all(pose["bodies"]["candidate"][:, :2] <= 1.0)


def test_preview_renderers_return_camera_and_skeleton_images():
    mapper = SkeletalPoseMapper()
    pose = make_pose()
    mapper.calibrate(pose)
    skeletal = mapper.update(pose, 100, (480, 640, 3))
    camera = np.zeros((480, 640, 3), dtype=np.uint8)

    camera_view = render_camera_view(camera, skeletal, "Tracking")
    skeleton_view = render_skeleton_view(skeletal, "Tracking")

    assert camera_view.shape == camera.shape
    assert skeleton_view.shape == (640, 640, 3)
    assert np.any(skeleton_view != (20, 24, 31))
