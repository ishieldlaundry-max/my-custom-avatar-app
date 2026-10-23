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
    pipe = NS(src_imgs=[frames[0]],
              src_infos=[[[{}, np.array([[2, 2], [6, 6]])]]], src_lmk_pre="stale")
    pipe.prepare_source = lambda *a, **k: True
    calls = []
    def render(frame, *args, **kwargs):
        assert pipe.src_lmk_pre is None
        calls.append((int(frame[0, 0, 0]), kwargs["first_frame"]))
        pipe.src_lmk_pre = "old landmarks"
        return frame, frame, frame, [{k: np.zeros((1, 1)) for k in ("pitch", "yaw", "roll")}]
    pipe.run = render
    namespace = dict(
        Fore=NS(RED=""), Style=NS(RESET_ALL=""),
        OmegaConf=NS(load=lambda _: NS(infer_params=NS(flag_pasteback=True))),
        FasterLivePortraitPipeline=lambda **_: pipe,
        resolve_camera_source=preview.resolve_camera_source,
        LiveTrackingPreview=preview.LiveTrackingPreview,
        LivePortraitEffects=effects.LivePortraitEffects,
        LatestFrameCapture=lambda _: reader, cv2=cv, np=np,
        os=__import__("os"), time=time, datetime=datetime, deque=deque,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "run.py", "exec"), namespace)
    namespace["run_with_video"](NS(
        cfg="unused", paste_back=True, animal=False, src_image="unused",
        realtime=True, dri_video="0", virtual_camera=False, realtime_metrics=False,
    ))
    assert calls == [(1, True), (2, False), (3, True)]
    reader.close.assert_called_once()
    assert any(call.args[0] == "Your camera" for call in cv.imshow.call_args_list)
