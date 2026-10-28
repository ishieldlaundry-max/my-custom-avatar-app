from pathlib import Path

from PIL import Image

from benchmark_detail_preview import (
    _landmark_crop,
    make_detail_sheet,
    sample_frame_indices,
)
from scripts.mimicmotion_detail_landmarks import serialize_pose


def _landmarks(center_x=0.5, center_y=0.5, count=68, confidence=0.9):
    points = [
        [
            center_x + ((index % 5) - 2) * 0.01,
            center_y + ((index % 7) - 3) * 0.01,
        ]
        for index in range(count)
    ]
    return {"points": points, "scores": [confidence] * count}


def test_detail_sample_indices_match_full_frame_timeline():
    assert sample_frame_indices(101) == [("15%", 15), ("50%", 50), ("85%", 85)]
    assert sample_frame_indices(1) == [("15%", 0), ("50%", 0), ("85%", 0)]
    assert sample_frame_indices(0) == []


def test_landmark_crop_requires_enough_confident_points():
    image = Image.new("RGB", (200, 160), color=(40, 80, 120))

    crop = _landmark_crop(
        image,
        _landmarks(count=68),
        min_points=20,
        scale=1.9,
    )
    unreliable = _landmark_crop(
        image,
        _landmarks(count=68, confidence=0.1),
        min_points=20,
        scale=1.9,
    )

    assert crop is not None
    assert crop.width > 10
    assert crop.height > 10
    assert unreliable is None


def test_detail_sheet_saves_confident_crops_and_skips_uncertain_regions(tmp_path):
    frame = Image.new("RGB", (200, 160), color=(90, 120, 150))
    sample_frames = [
        ("15%", 10, frame),
        ("50%", 50, frame),
        ("85%", 90, frame),
    ]
    detections = {
        "frames": [
            {
                "frame_index": 10,
                "people_count": 1,
                "face": _landmarks(),
                "hand_1": _landmarks(center_x=0.25, center_y=0.75, count=21),
                "hand_2": _landmarks(
                    center_x=0.75,
                    center_y=0.75,
                    count=21,
                    confidence=0.1,
                ),
            },
            {"frame_index": 50, "people_count": 2},
            {"frame_index": 90, "people_count": 0},
        ]
    }
    preview_path = tmp_path / "quality-details.png"

    path, crop_count = make_detail_sheet(
        sample_frames,
        detections,
        preview_path,
    )

    assert path == str(preview_path)
    assert crop_count == 2
    assert Image.open(preview_path).size == (960, 888)


def test_detail_sheet_is_omitted_when_no_landmarks_are_reliable(tmp_path):
    frame = Image.new("RGB", (200, 160), color=(90, 120, 150))
    path, crop_count = make_detail_sheet(
        [("15%", 0, frame)],
        {"frames": [{"frame_index": 0, "people_count": 2}]},
        tmp_path / "quality-details.png",
    )

    assert path is None
    assert crop_count == 0
    assert not (tmp_path / "quality-details.png").exists()


def test_landmark_helper_serializes_details_only_for_one_person():
    import numpy as np

    pose = {
        "bodies": {"candidate": np.zeros((1, 18, 2))},
        "faces": np.zeros((1, 68, 2)),
        "faces_score": np.ones((1, 68)),
        "hands": np.zeros((2, 21, 2)),
        "hands_score": np.ones((2, 21)),
    }
    result = serialize_pose(17, pose)

    assert result["frame_index"] == 17
    assert result["people_count"] == 1
    assert len(result["face"]["points"]) == 68
    assert len(result["hand_1"]["points"]) == 21
    assert len(result["hand_2"]["scores"]) == 21

    pose["bodies"]["candidate"] = np.zeros((2, 18, 2))
    ambiguous = serialize_pose(17, pose)
    assert ambiguous == {"frame_index": 17, "people_count": 2}
