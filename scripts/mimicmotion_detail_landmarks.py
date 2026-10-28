"""Detect face and hand landmarks in selected MimicMotion output frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def serialize_pose(frame_index: int, pose: dict) -> dict:
    people = pose.get("bodies", {}).get("candidate", [])
    person_count = len(people)
    result = {"frame_index": int(frame_index), "people_count": person_count}
    if person_count != 1:
        return result

    try:
        result.update(
            {
                "face": {
                    "points": _as_list(pose["faces"][0]),
                    "scores": _as_list(pose["faces_score"][0]),
                },
                "hand_1": {
                    "points": _as_list(pose["hands"][0]),
                    "scores": _as_list(pose["hands_score"][0]),
                },
                "hand_2": {
                    "points": _as_list(pose["hands"][1]),
                    "scores": _as_list(pose["hands_score"][1]),
                },
            }
        )
    except (IndexError, KeyError, TypeError):
        result = {"frame_index": int(frame_index), "people_count": person_count}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--backend-root", required=True, type=Path)
    parser.add_argument("--frame-indices", required=True)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    frame_indices = [int(value) for value in args.frame_indices.split(",") if value]
    if not frame_indices or any(index < 0 for index in frame_indices):
        raise ValueError("At least one non-negative frame index is required.")

    sys.path.insert(0, str(args.backend_root.resolve()))
    import decord
    from mimicmotion.dwpose.dwpose_detector import DWposeDetector

    model_dir = args.backend_root / "models" / "DWPose"
    detector = DWposeDetector(
        model_det=str(model_dir / "yolox_l.onnx"),
        model_pose=str(model_dir / "dw-ll_ucoco_384.onnx"),
        device="cpu",
    )
    reader = decord.VideoReader(str(args.video), ctx=decord.cpu(0))
    if any(index >= len(reader) for index in frame_indices):
        raise ValueError("A requested frame index is outside the rendered video.")

    results = []
    try:
        for index in frame_indices:
            frame = reader[index].asnumpy()
            results.append(serialize_pose(index, detector(frame)))
    finally:
        detector.release_memory()
    print(json.dumps({"frames": results}))


if __name__ == "__main__":
    main()
