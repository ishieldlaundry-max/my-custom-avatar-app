"""Measure an Alienware full-body render without changing render settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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
        "stream=duration,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        stream = json.loads(result.stdout).get("streams", [{}])[0]
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {}
    rate = stream.get("avg_frame_rate", "")
    fps = None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        if float(denominator):
            fps = round(float(numerator) / float(denominator), 3)
    return {
        "duration_seconds": float(stream["duration"]) if stream.get("duration") else None,
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "fps": fps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the 8 GB offline full-body render on Windows."
    )
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--driving-video", required=True, type=Path)
    parser.add_argument("--output-dir", default="results/full_body_benchmark", type=Path)
    parser.add_argument("--sample-seconds", default=1.0, type=float)
    args = parser.parse_args()

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
    ]
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
    report = {
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
        "video": _video_metadata(output) if output else {},
        "logs_tail": output_lines[-30:],
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Benchmark report: {report_path}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())