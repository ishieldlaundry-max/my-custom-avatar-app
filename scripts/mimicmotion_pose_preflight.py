"""Sparse, CPU-only DWPose Wholebody analysis for full-body preflight."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


REQUIRED_BODY_LANDMARKS = {
    0: "nose",
    2: "right shoulder",
    5: "left shoulder",
    4: "right wrist",
    7: "left wrist",
    8: "right hip",
    11: "left hip",
    9: "right knee",
    12: "left knee",
    10: "right ankle",
    13: "left ankle",
}

SHORT_CLIP_SECONDS = 6.0
MAX_SHORT_CLIP_SAMPLES = 24


def select_sample_indices(
    frame_count: int, fps: float | None, max_samples: int
) -> np.ndarray:
    """Sample short clips more densely while keeping CPU pose work capped."""
    if frame_count < 1:
        raise ValueError("The driver video contains no decodable frames.")
    if not 1 <= max_samples <= 12:
        raise ValueError("max_samples must be between 1 and 12.")

    sample_count = min(max_samples, frame_count)
    if (
        fps is not None
        and math.isfinite(float(fps))
        and float(fps) > 0
        and frame_count / float(fps) <= SHORT_CLIP_SECONDS
    ):
        sample_count = min(
            frame_count,
            MAX_SHORT_CLIP_SAMPLES,
            max_samples * 2,
        )

    return np.unique(
        np.linspace(0, frame_count - 1, num=sample_count, dtype=np.int64)
    )


def inspect_video(video_path: Path, backend_root: Path, max_samples: int) -> dict:
    started_at = time.perf_counter()
    sys.path.insert(0, str(backend_root))
    import decord
    import torch
    from mimicmotion.dwpose.dwpose_detector import DWposeDetector

    model_dir = backend_root / "models" / "DWPose"
    detector = DWposeDetector(
        model_det=str(model_dir / "yolox_l.onnx"),
        model_pose=str(model_dir / "dw-ll_ucoco_384.onnx"),
        device="cpu",
    )
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frame_count = len(reader)
    if frame_count < 1:
        raise ValueError("The driver video contains no decodable frames.")
    try:
        fps = float(reader.get_avg_fps())
    except (AttributeError, TypeError, ValueError):
        fps = None
    sample_indices = select_sample_indices(frame_count, fps, max_samples)

    sampled_results = []
    with torch.inference_mode():
        for sample_index, frame_index in enumerate(sample_indices, start=1):
            frame = reader[int(frame_index)].asnumpy()
            pose = detector(frame)
            subset = pose["bodies"]["subset"]
            people = []
            for person in subset:
                missing = [
                    name
                    for landmark_id, name in REQUIRED_BODY_LANDMARKS.items()
                    if landmark_id >= len(person) or person[landmark_id] < 0
                ]
                people.append({"missing_landmarks": missing})
            sampled_results.append(
                {"frame_index": int(frame_index), "people": people}
            )
            print(
                f"DWPose progress: {sample_index}/{len(sample_indices)}",
                file=sys.stderr,
                flush=True,
            )
    detector.release_memory()
    return {
        "sample_count": len(sampled_results),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "frames": sampled_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-driver", required=True, type=Path)
    parser.add_argument(
        "--max-samples",
        default=12,
        type=int,
        help="Long-clip DWPose sample cap; clips up to 6 seconds can use up to 24.",
    )
    args = parser.parse_args()
    if not args.inspect_driver.is_file():
        raise FileNotFoundError(f"Driver video not found: {args.inspect_driver}")
    if not 1 <= args.max_samples <= 12:
        raise ValueError("--max-samples must be between 1 and 12")
    report = inspect_video(args.inspect_driver, Path.cwd(), args.max_samples)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
