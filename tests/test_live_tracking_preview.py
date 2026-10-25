import ast
from collections import deque
import datetime
from pathlib import Path
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.utils import live_tracking_preview as preview
from src.utils import live_portrait_effects as effects
from src.utils.live_face_composite import (
    build_face_mask,
    build_face_mask_from_image_landmarks,
    composite_photo_face_on_driver,
)


@pytest.fixture
def ui(monkeypatch):
    mock = MagicMock()
    mock.WINDOW_NORMAL = 0
    mock.WND_PROP_VISIBLE = 4
    mock.getWindowProperty.return_value = 1
    mock.waitKey.return_value = -1
    monkeypatch.setattr(preview, "cv2", mock)
    return preview.LiveTrackingPreview(0), mock


def test_waits_for_deliberate_calibration_and_can_recalibrate(ui):
    controller, cv = ui
    assert not controller.ready
    cv.waitKey.return_value = ord("c")
    assert controller.poll()
    assert controller.ready and controller.needs_reference
    controller.tracking_result(False)
    assert controller.needs_reference
    assert "No face" in controller.status
    controller.tracking_result(True)
    assert not controller.needs_reference
    assert controller.ready
    controller.poll()
    assert controller.needs_reference


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_quit(ui, key):
    controller, cv = ui
    cv.waitKey.return_value = key
    assert not controller.poll()


def test_close_either_window_stops(ui):
    controller, cv = ui
    cv.getWindowProperty.side_effect = [1, 0]
    assert not controller.poll()


@pytest.mark.parametrize("key", ["c", "h", "s", "b", "l", "q"])
def test_buttons_dispatch_existing_actions(ui, key):
    controller, cv = ui
    controller.add_controls(np.zeros((480, 640, 3), np.uint8))
    bounds, _, label = next(b for b in controller.buttons if b[1] == ord(key))
    left, top, right, bottom = bounds
    controller._on_mouse(cv.EVENT_LBUTTONDOWN, (left + right) // 2,
                         (top + bottom) // 2, None, None)
    assert controller.poll() is (key != "q")
    assert controller.last_key == ord(key)
    if key == "c":
        assert controller.calibration_requested
    if key == "h":
        assert not controller.head_enabled


def test_clean_room_button_requests_background_capture(ui):
    controller, cv = ui
    controller.allow_background_capture = True
    controller.add_controls(np.zeros((480, 640, 3), np.uint8))
    bounds, _, _ = next(button for button in controller.buttons if button[1] == ord("g"))
    left, top, right, bottom = bounds

    controller._on_mouse(
        cv.EVENT_LBUTTONDOWN,
        (left + right) // 2,
        (top + bottom) // 2,
        None,
        None,
    )

    assert controller.poll()
    assert controller.background_capture_requested
    assert "Stay out of frame" in controller.status


def test_clean_room_capture_uses_a_median_of_camera_frames(ui, monkeypatch):
    controller, _ = ui
    controller.show_camera = MagicMock()
    controller.poll = MagicMock(return_value=True)
    frames = [np.full((4, 4, 3), value, np.uint8) for value in (10, 240, 20)]
    capture = NS(read=MagicMock(side_effect=[(True, frames[1]), (True, frames[2])]))
    captured = {}

    def median(frame_list, target_shape):
        captured["frames"] = frame_list
        captured["target_shape"] = target_shape
        return np.median(np.stack(frame_list), axis=0).astype(np.uint8)

    monkeypatch.setattr(preview, "median_clean_background", median)
    background = controller.capture_clean_background(
        capture,
        frames[0],
        (4, 4),
        sample_count=3,
        sample_interval=0,
    )

    assert len(captured["frames"]) == 3
    assert captured["target_shape"] == (4, 4)
    assert np.all(background == 20)
    assert capture.read.call_count == 2


def test_toolbar_click_does_not_select_shoulder(ui):
    controller, cv = ui
    controller.effects = MagicMock()
    controller.add_controls(np.zeros((360, 640, 3), np.uint8))
    controller._on_mouse(cv.EVENT_LBUTTONDOWN, 20, 375, None, None)
    controller.effects._driver_click.assert_not_called()
    controller._on_mouse(cv.EVENT_LBUTTONDOWN, 50, 200, None, None)
    controller.effects._driver_click.assert_called_once()


def test_face_swap_preview_explains_camera_body_and_hides_shoulder_controls(monkeypatch):
    cv = MagicMock()
    monkeypatch.setattr(preview, "cv2", cv)
    controller = preview.LiveTrackingPreview(0, face_swap_mode=True)

    controller.add_controls(np.zeros((480, 640, 3), np.uint8))
    labels = [label for _, _, label in controller.buttons]
    controller.tracking_result(True)

    assert labels == ["Calibrate face (C)", "Head: ON (H)", "Stop tracking (Q)"]
    assert "webcam body/background retained" in controller.status
    controller.tracking_result(False)
    assert "webcam body/background remain visible" in controller.status


def test_buttons_follow_camera_height_and_show_toggle_state(ui):
    controller, cv = ui
    controller.head_enabled = False
    image = controller.add_controls(np.zeros((720, 640, 3), np.uint8))
    assert image.shape == (852, 640, 3)
    assert all(bounds[1] >= 720 for bounds, _, _ in controller.buttons)
    assert any("Head: OFF" in label for _, _, label in controller.buttons)


def test_source_is_explicit(tmp_path):
    assert preview.resolve_camera_source("0") == (True, 0)
    assert preview.resolve_camera_source("2") == (True, 2)
    with pytest.raises(FileNotFoundError):
        preview.resolve_camera_source(str(tmp_path / "missing.mp4"))
    clip = tmp_path / "clip.mp4"
    clip.touch()
    assert preview.resolve_camera_source(str(clip)) == (False, str(clip))


def test_live_loop_uses_current_frame_and_fresh_detection(monkeypatch, tmp_path):
    """Exercise the real loop with fake devices/models, without loading CUDA."""
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    module = ast.parse((root / "run.py").read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == "run_with_video")
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(5)]
    reader = MagicMock()
    reader.start.return_value = reader
    reader.read.side_effect = [(True, f) for f in frames]
    cv = MagicMock()
    cv.VideoCapture.return_value.get.return_value = 30
    cv.getWindowProperty.return_value = 1
    # First frame awaits calibration; then calibrate, track, recalibrate, quit.
    cv.waitKey.side_effect = [-1, ord("c"), -1, ord("c"), ord("q")]
    cv.flip.side_effect = lambda f, _: f.copy()
    cv.resize.side_effect = lambda f, _: f
    cv.cvtColor.side_effect = lambda f, _: f.copy()
    monkeypatch.setattr(preview, "cv2", cv)
    monkeypatch.setattr(effects, "cv2", cv)
    source_landmarks = np.array(
        [
            [256, 90], [325, 115], [370, 180], [360, 270],
            [320, 345], [256, 370], [192, 345], [152, 270],
            [142, 180], [187, 115],
        ],
        dtype=np.float32,
    )
    pipe = NS(
        src_imgs=[frames[0]],
        src_infos=[[[{}, source_landmarks, np.eye(3, dtype=np.float32)]]],
        src_lmk_pre="stale",
        is_source_video=False,
        last_driver_crop_to_original=np.array(
            [[1 / 64, 0, 0], [0, 1 / 64, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
    )
    pipe.prepare_source = lambda *a, **k: True
    calls = []
    def render(frame, *args, **kwargs):
        assert pipe.src_lmk_pre is None
        calls.append((int(frame[0, 0, 0]), kwargs["first_frame"]))
        pipe.src_lmk_pre = "old landmarks"
        pipe.last_driver_crop_to_original = np.array(
            [[1 / 64, 0, 0], [0, 1 / 64, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        photo_face = np.zeros((512, 512, 3), dtype=np.uint8)
        photo_face[:] = (255, 0, 0)
        return (
            frame,
            photo_face,
            frame,
            [{k: np.zeros((1, 1)) for k in ("pitch", "yaw", "roll")}],
        )
    pipe.run = render
    config = NS(infer_params=NS(flag_pasteback=True))
    namespace = dict(
        Fore=NS(RED=""), Style=NS(RESET_ALL=""),
        OmegaConf=NS(load=lambda _: config),
        FasterLivePortraitPipeline=lambda **_: pipe,
        resolve_camera_source=preview.resolve_camera_source,
        LiveTrackingPreview=preview.LiveTrackingPreview,
        LivePortraitEffects=effects.LivePortraitEffects,
        build_face_mask_from_image_landmarks=build_face_mask_from_image_landmarks,
        composite_photo_face_on_driver=composite_photo_face_on_driver,
        LatestFrameCapture=lambda _: reader, cv2=cv, np=np,
        os=__import__("os"), time=time, datetime=datetime, deque=deque,
        DriverBackgroundError=RuntimeError,
        extract_target_alpha=lambda *_: pytest.fail(
            "Live webcam face swap must not extract or composite the target body."
        ),
        composite_target_over_driver_background=lambda *_: pytest.fail(
            "Live webcam face swap must preserve the camera body/background."
        ),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "run.py", "exec"), namespace)
    namespace["run_with_video"](NS(
        cfg="unused", paste_back=True, animal=False, src_image="unused",
        realtime=True, dri_video="0", virtual_camera=False, realtime_metrics=False,
        use_driver_background=True,
    ))
    assert calls == [(1, True), (2, False), (3, True)]
    assert config.infer_params.flag_eye_retargeting is True
    assert config.infer_params.flag_lip_retargeting is True
    reader.close.assert_called_once()
    assert any(call.args[0] == "Your camera" for call in cv.imshow.call_args_list)
    render_frames = [
        call.args[1]
        for call in cv.imshow.call_args_list
        if call.args[0] == "Render"
    ]
    np.testing.assert_array_equal(render_frames[-1][0, 0], [3, 3, 3])
    np.testing.assert_array_equal(render_frames[-1][4, 4], [0, 0, 255])
