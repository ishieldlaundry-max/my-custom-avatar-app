"""CLI entry point for offline full-body avatar animation."""

import argparse
from pathlib import Path

from src.pipelines.full_body_animation_pipeline import (
    FullBodyAnimationPipeline,
    FullBodyRenderOptions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a full-body target image from a recorded pose-driving video."
    )
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--driving-video", required=True, type=Path)
    parser.add_argument("--output-dir", default="results/full_body", type=Path)
    parser.add_argument("--resolution", default=512, type=int, choices=(512, 576))
    parser.add_argument("--steps", default=15, type=int)
    parser.add_argument("--fps", default=15, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    options = FullBodyRenderOptions(
        resolution=args.resolution,
        num_inference_steps=args.steps,
        fps=args.fps,
        seed=args.seed,
    )
    output = FullBodyAnimationPipeline().render(
        args.source_image,
        args.driving_video,
        args.output_dir,
        options,
    )
    print(output)


if __name__ == "__main__":
    main()