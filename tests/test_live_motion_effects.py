import numpy as np
import cv2

from src.utils.live_motion import AdaptiveMotionFilter, bounded_pose
from src.utils.live_portrait_effects import (
    ShoulderFlow, deform_shoulders, shade_portrait, LivePortraitEffects,
)


def motion(value=0):
    return {k: np.full((1, 1), value, dtype=np.float32)
            for k in ("exp", "pitch", "yaw", "roll")}


def test_filter_reduces_small_jitter_and_resets():
    smoother = AdaptiveMotionFilter()
    smoother.process(motion(), now=0)
    result = smoother.process(motion(.001), now=.033)
    assert 0 < result["exp"].item() < .001
    np.testing.assert_array_equal(
        smoother.process(motion(.5), reset=True, now=.066)["exp"], [[.5]])


def test_expression_change_has_less_smoothing_than_tiny_jitter():
    a, b = AdaptiveMotionFilter(), AdaptiveMotionFilter()
    a.process(motion(), now=0)
    b.process(motion(), now=0)
    small = a.process(motion(.001), now=.033)["exp"].item() / .001
    large = b.process(motion(.1), now=.033)["exp"].item() / .1
    assert large > small
    assert large > .9


def test_pose_is_relative_bounded_and_neutral_at_calibration():
    assert all(np.all(x == 0) for x in bounded_pose(motion(20), motion(20)))
    assert [x.item() for x in bounded_pose(motion(100), motion(20))] == [10, 15, 10]
    assert [x.item() for x in bounded_pose(motion(-100), motion(20))] == [-10, -15, -10]


def test_shoulder_flow_tracks_real_image_translation():
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    tracker = ShoulderFlow()
    assert tracker.reset(frame, [(.3, .6), (.7, .6)])
    translated = cv2.warpAffine(frame, np.float32([[1, 0, 3], [0, 1, 2]]), (300, 200))
    offsets = tracker.update(translated)
    np.testing.assert_allclose(offsets, [[.01, .01], [.01, .01]], atol=.002)


def test_textureless_shoulders_do_not_claim_tracking():
    tracker = ShoulderFlow()
    frame = np.full((200, 300, 3), 128, np.uint8)
    assert not tracker.reset(frame, [(.3, .6), (.7, .6)])
    assert tracker.update(frame) is None


def test_shoulder_warp_preserves_head_and_neutral_frame():
    image = np.random.default_rng(4).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    anchors = [(.25, .7), (.75, .7)]
    neutral = deform_shoulders(image, anchors, np.zeros((2, 2)))
    np.testing.assert_array_equal(neutral, image)
    moved = deform_shoulders(image, anchors, [[.03, -.03], [-.03, .02]])
    np.testing.assert_array_equal(moved[:100], image[:100])
    assert np.any(moved[130:] != image[130:])
    assert moved.dtype == np.uint8


def test_shading_is_neutral_bounded_and_localized():
    image = np.full((200, 200, 3), 128, np.uint8)
    bounds = (50, 30, 150, 130)
    np.testing.assert_array_equal(shade_portrait(image, bounds, (0, 0, 0)), image)
    shaded = shade_portrait(image, bounds, (10, 15, 0))
    assert np.any(shaded != image)
    assert np.max(np.abs(shaded.astype(float) - image)) < 11
    np.testing.assert_array_equal(shaded[180:], image[180:])


def test_shoulder_selection_and_effect_toggles(monkeypatch):
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a: None)
    fx = LivePortraitEffects(np.zeros((300, 400, 3), np.uint8), [[100, 20], [300, 150]])
    fx.handle_key(ord("s"))
    fx._driver_click(cv2.EVENT_LBUTTONDOWN, 100, 300, None, None)
    fx._driver_click(cv2.EVENT_LBUTTONDOWN, 500, 300, None, None)
    fx._target_click(cv2.EVENT_LBUTTONDOWN, 60, 200, None, None)
    fx._target_click(cv2.EVENT_LBUTTONDOWN, 340, 200, None, None)
    assert not fx.selecting
    assert fx.shoulders_enabled
    fx.handle_key(ord("b"))
    assert not fx.shoulders_enabled
    fx.handle_key(ord("l"))
    assert not fx.shadows_enabled
