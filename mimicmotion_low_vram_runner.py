"""Run the upstream MimicMotion inference with Diffusers CPU offload enabled.

This file intentionally imports the isolated checkout at runtime. It should be
executed by third_party/MimicMotion/.venv, never by the face-pipeline Python
environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # The backend checkout is the working directory, but this runner lives in
    # the application repository. Import upstream modules only after adding
    # that checkout to sys.path.
    backend_root = Path.cwd()
    sys.path.insert(0, str(backend_root))

    from inference import preprocess, run_pipeline  # noqa: PLC0415
    from mimicmotion.utils.geglu_patch import patch_geglu_inplace  # noqa: PLC0415
    from mimicmotion.utils.loader import create_pipeline  # noqa: PLC0415
    from mimicmotion.utils.utils import save_to_mp4  # noqa: PLC0415

    patch_geglu_inplace()
    logging.basicConfig(level=logging.INFO)
    if not os.environ.get("MIMICMOTION_CPU_OFFLOAD", "1") == "0":
        logging.info("Enabling Diffusers sequential CPU offload for the 8 GB preset.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MimicMotion full-body rendering requires CUDA on the Alienware."
        )
    torch.set_default_dtype(torch.float16)
    device = torch.device("cuda")
    config = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(config, device)
    if os.environ.get("MIMICMOTION_CPU_OFFLOAD", "1") != "0":
        pipeline.enable_model_cpu_offload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in config.test_case:
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path,
            task.ref_image_path,
            resolution=task.resolution,
            sample_stride=task.sample_stride,
        )
        frames = run_pipeline(pipeline, image_pixels, pose_pixels, device, task)
        save_to_mp4(
            frames,
            str(
                output_dir
                / f"{Path(task.ref_video_path).stem}_{datetime.now():%Y%m%d%H%M%S}.mp4"
            ),
            fps=task.fps,
        )


if __name__ == "__main__":
    main()