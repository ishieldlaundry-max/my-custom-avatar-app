#!/usr/bin/env python3
"""Generate the small neutral-motion fixture used by the API pickle test.

The fixture intentionally contains one valid, neutral animal-motion frame. It
does not require model checkpoints, a GPU, or a face detector, so it can be
regenerated in a clean checkout.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def build_fixture() -> dict:
    """Return the motion-template shape consumed by run_with_pkl()."""
    return {
        "n_frames": 1,
        "output_fps": 1,
        "motion": [
            {
                "R": np.eye(3, dtype=np.float32)[None, ...],
                "exp": np.zeros((1, 21, 3), dtype=np.float32),
                "t": np.zeros((1, 3), dtype=np.float32),
                "scale": np.ones((1, 1), dtype=np.float32),
                "kp": np.zeros((1, 21, 3), dtype=np.float32),
            }
        ],
        "c_eyes_lst": [np.zeros((1, 1), dtype=np.float32)],
        "c_lip_lst": [np.zeros((1, 1), dtype=np.float32)],
    }


def write_fixture(output: Path) -> None:
    """Write the deterministic fixture to output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(build_fixture(), handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("assets/examples/driving/d8.pkl"),
        help="output pickle path (default: assets/examples/driving/d8.pkl)",
    )
    args = parser.parse_args()

    write_fixture(args.output)
    print(f"Wrote neutral motion fixture to {args.output}")


if __name__ == "__main__":
    main()