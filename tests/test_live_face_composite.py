from pathlib import Path

import numpy as np
import pytest

from src.utils.live_face_composite import (
    build_face_mask,
    build_face_mask_from_image_landmarks,
    composite_photo_face_on_driver,
)


def test_landmark_mask_is_a_soft_face_shape_not_a_full_crop():
    landmarks = np.array(
        [
            [16, 3], [22, 5], [27, 10], [29, 16], [27, 22], [22, 27],
            [16, 29], [10, 27], [5, 22], [3, 16], [5, 10], [10, 5],
        ],
        dtype=np.float32,
    )

    mask = build_face_mask(landmarks, (32, 32))

    assert mask.shape == (32, 32)
    assert mask.dtype == np.float32
    assert mask[16, 16] > 0.95
    assert mask[0, 0] < 0.05
    assert np.all((mask >= 0) & (mask <= 1))
    assert np.any((mask > 0) & (mask < 1))


def test_source_image_landmarks_are_mapped_into_crop_before_masking():
    landmarks_in_image = np.array(
        [[116, 216], [132, 216], [132, 232], [116, 232]],
        dtype=np.float32,
    )
    crop_to_image = np.array(
        [[2.0, 0.0, 100.0], [0.0, 2.0, 200.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    mask = build_face_mask_from_image_landmarks(
        landmarks_in_image,
        crop_to_image,
        (32, 32),
    )

    assert mask[12, 12] > 0.95
    assert mask[0, 0] < 0.05
    assert mask[12, 24] < 0.05


def test_face_composite_preserves_webcam_body_and_background():
    driver = np.full((16, 16, 3), (20, 40, 60), dtype=np.uint8)  # BGR
    photo_face = np.full((16, 16, 3), (230, 20, 10), dtype=np.uint8)  # RGB
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[5:11, 5:11] = 1.0

    result = composite_photo_face_on_driver(
        driver,
        photo_face,
        mask,
        np.eye(3, dtype=np.float32),
        color_match_strength=0.0,
    )

    np.testing.assert_array_equal(result[0, 0], driver[0, 0])
    np.testing.assert_array_equal(result[8, 8], [10, 20, 230])
    assert result.shape == driver.shape


def test_face_composite_maps_only_the_face_crop_to_its_camera_location():
    driver = np.full((24, 24, 3), (10, 20, 30), dtype=np.uint8)
    photo_face = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)  # RGB red
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    crop_to_driver = np.array(
        [[1.0, 0.0, 8.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    result = composite_photo_face_on_driver(
        driver,
        photo_face,
        mask,
        crop_to_driver,
        color_match_strength=0.0,
    )

    np.testing.assert_array_equal(result[0, 0], driver[0, 0])
    np.testing.assert_array_equal(result[12, 11], [0, 0, 255])
    np.testing.assert_array_equal(result[12, 20], driver[12, 20])


def test_face_composite_keeps_mask_and_crop_transform_resolution():
    driver = np.full((20, 20, 3), (10, 20, 30), dtype=np.uint8)
    photo_face = np.zeros((4, 4, 3), dtype=np.uint8)
    photo_face[:, :, 0] = 255  # RGB red
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    crop_to_driver = np.array(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    result = composite_photo_face_on_driver(
        driver,
        photo_face,
        mask,
        crop_to_driver,
        color_match_strength=0.0,
    )

    # The mask/transform describe an 8x8 crop; the 4x4 generated output must
    # be enlarged into that coordinate space before being mapped to the frame.
    np.testing.assert_array_equal(result[10, 10], [0, 0, 255])
    np.testing.assert_array_equal(result[5, 5], driver[5, 5])


def test_face_composite_matches_live_camera_color_in_the_face_crop():
    driver = np.full((16, 16, 3), (60, 80, 100), dtype=np.uint8)  # BGR
    photo_face = np.full((16, 16, 3), (200, 160, 120), dtype=np.uint8)  # RGB
    mask = np.ones((16, 16), dtype=np.float32)

    result = composite_photo_face_on_driver(
        driver,
        photo_face,
        mask,
        np.eye(3, dtype=np.float32),
        color_match_strength=1.0,
    )

    np.testing.assert_allclose(result[8, 8], driver[8, 8], atol=2)


def test_live_pipeline_exposes_the_driver_crop_to_camera_transform():
    source = Path("src/pipelines/faster_live_portrait_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "self.last_driver_crop_to_original = ret_dct[\"M_c2o\"]" in source
    assert "self.last_driver_crop_to_original = None" in source
    assert "self.last_driver_landmarks_in_crop = np.asarray(" in source


@pytest.mark.parametrize(
    "landmarks",
    [
        np.array([[1, 1], [2, 2]], dtype=np.float32),
        np.array([[1, 1], [np.nan, 2], [2, 2]], dtype=np.float32),
    ],
)
def test_face_mask_rejects_insufficient_valid_landmarks(landmarks):
    with pytest.raises(ValueError):
        build_face_mask(landmarks, (8, 8))
