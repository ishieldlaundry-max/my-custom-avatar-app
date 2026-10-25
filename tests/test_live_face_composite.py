from pathlib import Path

import numpy as np
import pytest

from src.utils.live_face_composite import (
    build_face_mask,
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
    )

    np.testing.assert_array_equal(result[0, 0], driver[0, 0])
    np.testing.assert_array_equal(result[12, 11], [0, 0, 255])
    np.testing.assert_array_equal(result[12, 20], driver[12, 20])


def test_live_pipeline_exposes_the_driver_crop_to_camera_transform():
    source = Path("src/pipelines/faster_live_portrait_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "self.last_driver_crop_to_original = ret_dct[\"M_c2o\"]" in source
    assert "self.last_driver_crop_to_original = None" in source


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
