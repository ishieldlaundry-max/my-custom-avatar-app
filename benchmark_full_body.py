"""Measure and compare Alienware full-body render settings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_detail_preview import (
    SAMPLE_LOCATIONS,
    make_quality_detail_preview,
)


def _gpu_sample() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 4:
        return {"error": f"Unexpected nvidia-smi output: {result.stdout.strip()}"}
    return {
        "name": fields[0],
        "memory_used_mb": int(fields[1]),
        "memory_total_mb": int(fields[2]),
        "utilization_percent": int(fields[3]),
    }


def _video_metadata(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
        stream = probe.get("streams", [{}])[0]
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {}
    rate = stream.get("avg_frame_rate", "")
    fps = None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        try:
            denominator_value = float(denominator)
            numerator_value = float(numerator)
        except ValueError:
            denominator_value = 0
            numerator_value = 0
        if denominator_value:
            fps = round(numerator_value / denominator_value, 3)
    duration = stream.get("duration") or probe.get("format", {}).get("duration")
    frames = stream.get("nb_frames")
    try:
        duration = float(duration) if duration not in (None, "N/A") else None
    except (TypeError, ValueError):
        duration = None
    try:
        frames = int(frames) if frames not in (None, "N/A") else None
    except (TypeError, ValueError):
        frames = None
    return {"duration_seconds": duration, "frames": frames, "fps": fps}


def _duration_comparison(driver: dict, output: dict) -> dict:
    """Compare the rendered clip duration with its recorded motion source."""
    driver_duration = driver.get("duration_seconds")
    output_duration = output.get("duration_seconds")
    output_fps = output.get("fps")
    if driver_duration is None or output_duration is None:
        return {
            "driver_duration_seconds": driver_duration,
            "output_duration_seconds": output_duration,
            "delta_seconds": None,
            "duration_ratio": None,
            "matches_driver_timing": None,
            "tolerance_seconds": None,
        }
    delta = output_duration - driver_duration
    tolerance = max(0.1, 2 / output_fps) if output_fps and output_fps > 0 else 0.1
    return {
        "driver_duration_seconds": driver_duration,
        "output_duration_seconds": output_duration,
        "delta_seconds": round(delta, 3),
        "duration_ratio": round(output_duration / driver_duration, 4)
        if driver_duration > 0
        else None,
        "matches_driver_timing": abs(delta) <= tolerance,
        "tolerance_seconds": round(tolerance, 3),
    }


def _make_quality_preview(video_path: Path | None, preview_path: Path) -> str | None:
    """Save three evenly spaced full-frame samples for manual quality review."""
    if video_path is None:
        return None
    try:
        import cv2
        from PIL import Image, ImageDraw

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return None
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                return None
            cards = []
            for label, fraction in SAMPLE_LOCATIONS:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    min(frame_count - 1, round((frame_count - 1) * fraction)),
                )
                success, frame = capture.read()
                if not success:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((480, 560))
                cards.append((label, image.copy()))
        finally:
            capture.release()
        if not cards:
            return None

        card_width = 500
        card_height = 600
        sheet = Image.new(
            "RGB",
            (card_width * len(cards), card_height),
            color=(12, 18, 28),
        )
        draw = ImageDraw.Draw(sheet)
        for index, (label, image) in enumerate(cards):
            x = index * card_width
            draw.text((x + 12, 8), f"Driver timeline {label}", fill=(235, 245, 255))
            image_x = x + (card_width - image.width) // 2
            image_y = 32 + (560 - image.height) // 2
            sheet.paste(image, (image_x, image_y))
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(preview_path)
        return str(preview_path)
    except (ImportError, OSError, ValueError):
        return None


def _run_step_comparison(args: argparse.Namespace) -> int:
    """Run each step setting repeatedly and preserve every individual report."""
    steps = args.compare_steps
    repetitions = getattr(args, "repetitions", 1)
    if steps[0] == steps[1]:
        raise ValueError("Choose two different denoising-step values to compare.")
    if any(step < 1 or step > 50 for step in steps):
        raise ValueError("Denoising steps must be between 1 and 50.")
    if not 1 <= repetitions <= 5:
        raise ValueError("Repetitions must be between 1 and 5.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    return_codes = []
    runs_by_steps = {step_count: [] for step_count in steps}
    for repetition in range(1, repetitions + 1):
        for step_count in steps:
            run_dir = (
                args.output_dir
                / f"steps_{step_count}"
                / f"repetition_{repetition:02d}"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source-image",
                str(args.source_image),
                "--driving-video",
                str(args.driving_video),
                "--output-dir",
                str(run_dir),
                "--resolution",
                str(args.resolution),
                "--steps",
                str(step_count),
                "--sample-seconds",
                str(args.sample_seconds),
            ]
            if args.fps is not None:
                command.extend(["--fps", str(args.fps)])
            result = subprocess.run(command, check=False)
            return_codes.append(result.returncode)
            report_path = run_dir / "benchmark.json"
            try:
                report = (
                    json.loads(report_path.read_text(encoding="utf-8"))
                    if report_path.is_file()
                    else {}
                )
            except json.JSONDecodeError:
                report = {}
            run = {
                **report,
                "steps": step_count,
                "repetition": repetition,
                "run_dir": str(run_dir),
                "return_code": result.returncode,
            }
            runs.append(run)
            runs_by_steps[step_count].append(run)

    settings = []
    for step_count in steps:
        setting_runs = runs_by_steps[step_count]
        successful_wall_times = [
            float(run["wall_seconds"])
            for run in setting_runs
            if run.get("return_code") == 0
            and isinstance(run.get("wall_seconds"), (int, float))
            and math.isfinite(float(run["wall_seconds"]))
        ]
        peak_memory_values = [
            int(run["peak_memory_used_mb"])
            for run in setting_runs
            if isinstance(run.get("peak_memory_used_mb"), (int, float))
            and math.isfinite(float(run["peak_memory_used_mb"]))
        ]
        settings.append(
            {
                "steps": step_count,
                "repetitions_requested": repetitions,
                "successful_runs": sum(
                    run.get("return_code") == 0 for run in setting_runs
                ),
                "median_wall_seconds": (
                    round(statistics.median(successful_wall_times), 3)
                    if successful_wall_times
                    else None
                ),
                "highest_peak_memory_used_mb": (
                    max(peak_memory_values) if peak_memory_values else None
                ),
                "runs": setting_runs,
            }
        )

    comparison = {
        "resolution": args.resolution,
        "comparison": "Denoising steps at fixed resolution",
        "repetitions_per_setting": repetitions,
        "settings": settings,
        "runs": runs,
        "visual_review": {
            "contact_sheets": [
                run.get("quality_preview") for run in runs if run.get("quality_preview")
            ],
            "detail_sheets": [
                run.get("quality_detail_preview")
                for run in runs
                if run.get("quality_detail_preview")
            ],
            "status": "notes recorded" if args.visual_notes else "manual review pending",
            "notes": args.visual_notes or None,
            "guidance": (
                "Use the full-frame sheets to compare composition and motion, and "
                "the detail sheets to inspect detected faces and hands. Detail "
                "crops are not quality scores."
            ),
        },
    }
    report_path = args.output_dir / "quality-comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Quality comparison report: {report_path}")
    return max(return_codes, default=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the 8 GB offline full-body render on Windows."
    )
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--driving-video", required=True, type=Path)
    parser.add_argument("--output-dir", default="results/full_body_benchmark", type=Path)
    parser.add_argument(
        "--resolution", default=512, type=int, choices=(384, 512, 576)
    )
    parser.add_argument("--steps", default=15, type=int)
    parser.add_argument(
        "--compare-steps",
        nargs=2,
        type=int,
        metavar=("BASELINE", "CANDIDATE"),
        help="Run two denoising-step settings sequentially and save a comparison report.",
    )
    parser.add_argument(
        "--repetitions",
        default=1,
        type=int,
        help="Runs per setting with --compare-steps (1-5; default: 1).",
    )
    parser.add_argument(
        "--visual-notes",
        default="",
        help="Optional manual observations to save in quality-comparison.json.",
    )
    parser.add_argument(
        "--fps",
        default=None,
        type=float,
        help="Explicit output FPS; omit to preserve the driver's original timing.",
    )
    parser.add_argument("--sample-seconds", default=1.0, type=float)
    args = parser.parse_args()

    if not 1 <= args.repetitions <= 5:
        parser.error("--repetitions must be between 1 and 5.")
    if args.repetitions > 1 and not args.compare_steps:
        parser.error("--repetitions requires --compare-steps.")
    if args.compare_steps:
        return _run_step_comparison(args)

    if not 1 <= args.steps <= 50:
        parser.error("--steps must be between 1 and 50.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "full_body.py",
        "--source-image",
        str(args.source_image),
        "--driving-video",
        str(args.driving_video),
        "--output-dir",
        str(args.output_dir),
        "--resolution",
        str(args.resolution),
        "--steps",
        str(args.steps),
    ]
    if args.fps is not None:
        command.extend(["--fps", str(args.fps)])
    started = time.monotonic()
    started_epoch = time.time()
    log_path = args.output_dir / "benchmark-run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        samples = []
        while process.poll() is None:
            sample = _gpu_sample()
            sample["elapsed_seconds"] = round(time.monotonic() - started, 3)
            samples.append(sample)
            time.sleep(max(0.1, args.sample_seconds))
        process.wait()
    output_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    wall_seconds = round(time.monotonic() - started, 3)

    outputs = [
        path
        for path in args.output_dir.rglob("*.mp4")
        if path.stat().st_mtime >= started_epoch
    ]
    output = max(outputs, key=lambda path: path.stat().st_mtime) if outputs else None
    valid_gpu_samples = [sample for sample in samples if "memory_used_mb" in sample]
    driver_metadata = _video_metadata(args.driving_video)
    output_metadata = _video_metadata(output) if output else {}
    quality_preview = _make_quality_preview(
        output,
        args.output_dir / "quality-samples.png",
    )
    quality_detail_preview, quality_detail_note = (
        make_quality_detail_preview(
            output,
            args.output_dir / "quality-details.png",
        )
        if output
        else (None, "Detail crops were not created because no output video was found.")
    )
    report = {
        "resolution": args.resolution,
        "steps": args.steps,
        "return_code": process.returncode,
        "wall_seconds": wall_seconds,
        "gpu": valid_gpu_samples[0].get("name") if valid_gpu_samples else None,
        "peak_memory_used_mb": max(
            (sample["memory_used_mb"] for sample in valid_gpu_samples),
            default=None,
        ),
        "gpu_memory_total_mb": max(
            (sample["memory_total_mb"] for sample in valid_gpu_samples),
            default=None,
        ),
        "output": str(output) if output else None,
        "driver": driver_metadata,
        "video": output_metadata,
        "timing": _duration_comparison(driver_metadata, output_metadata),
        "requested_output_fps": args.fps,
        "quality_preview": quality_preview,
        "quality_detail_preview": quality_detail_preview,
        "quality_detail_note": quality_detail_note,
        "logs_tail": output_lines[-30:],
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Benchmark report: {report_path}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())