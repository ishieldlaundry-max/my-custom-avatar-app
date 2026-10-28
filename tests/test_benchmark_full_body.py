import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from benchmark_full_body import (
    _duration_comparison,
    _make_quality_preview,
    _run_step_comparison,
)


def test_benchmark_reports_matching_three_second_30fps_driver():
    comparison = _duration_comparison(
        {"duration_seconds": 3.0, "frames": 90, "fps": 30.0},
        {"duration_seconds": 3.0, "frames": 90, "fps": 30.0},
    )

    assert comparison["matches_driver_timing"] is True
    assert comparison["driver_duration_seconds"] == 3.0
    assert comparison["output_duration_seconds"] == 3.0
    assert comparison["delta_seconds"] == 0
    assert comparison["duration_ratio"] == 1.0


def test_benchmark_flags_six_second_output_for_three_second_driver():
    comparison = _duration_comparison(
        {"duration_seconds": 3.0, "frames": 90, "fps": 30.0},
        {"duration_seconds": 6.0, "frames": 90, "fps": 15.0},
    )

    assert comparison["matches_driver_timing"] is False
    assert comparison["delta_seconds"] == 3.0
    assert comparison["duration_ratio"] == 2.0


def test_quality_preview_saves_three_frame_contact_sheet(tmp_path, monkeypatch):
    class Capture:
        def isOpened(self):
            return True

        def get(self, _property):
            return 90

        def set(self, _property, _value):
            return True

        def read(self):
            return True, np.zeros((32, 24, 3), dtype=np.uint8)

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda _path: Capture(),
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_POS_FRAMES=2,
        COLOR_BGR2RGB=3,
        cvtColor=lambda frame, _conversion: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    preview_path = tmp_path / "quality-samples.png"

    result = _make_quality_preview(tmp_path / "render.mp4", preview_path)

    assert result == str(preview_path)
    assert Image.open(preview_path).size == (1500, 600)


def test_step_comparison_runs_each_setting_and_saves_summary(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, check):
        assert check is False
        commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        steps = int(command[command.index("--steps") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "render.mp4").write_bytes(b"render")
        (output_dir / "quality-samples.png").write_bytes(b"sample")
        (output_dir / "benchmark.json").write_text(
            json.dumps(
                {
                    "steps": steps,
                    "wall_seconds": 120 + steps,
                    "peak_memory_used_mb": 7000 + steps,
                    "return_code": 0,
                    "output": str(output_dir / "render.mp4"),
                    "quality_preview": str(output_dir / "quality-samples.png"),
                    "quality_detail_preview": str(
                        output_dir / "quality-details.png"
                    ),
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("benchmark_full_body.subprocess.run", fake_run)
    args = Namespace(
        compare_steps=[15, 20],
        output_dir=tmp_path / "comparison",
        source_image=Path("target.png"),
        driving_video=Path("driver.mp4"),
        resolution=384,
        sample_seconds=0.25,
        fps=None,
        repetitions=1,
        visual_notes="Candidate looks sharper.",
    )

    assert _run_step_comparison(args) == 0
    report = json.loads(
        (args.output_dir / "quality-comparison.json").read_text(encoding="utf-8")
    )
    assert len(commands) == 2
    assert [run["steps"] for run in report["runs"]] == [15, 20]
    assert [run["peak_memory_used_mb"] for run in report["runs"]] == [7015, 7020]
    assert len(report["visual_review"]["contact_sheets"]) == 2
    assert len(report["visual_review"]["detail_sheets"]) == 2
    assert "not quality scores" in report["visual_review"]["guidance"]
    assert report["visual_review"]["notes"] == "Candidate looks sharper."
    assert [setting["median_wall_seconds"] for setting in report["settings"]] == [
        135.0,
        140.0,
    ]
    assert [setting["highest_peak_memory_used_mb"] for setting in report["settings"]] == [
        7015,
        7020,
    ]
    assert all(Path(run["output"]).is_file() for run in report["runs"])


def test_step_comparison_preserves_each_repetition_and_aggregates_median_and_vram(
    tmp_path,
    monkeypatch,
):
    commands = []
    wall_times = {
        15: [100.0, 140.0, 120.0],
        20: [110.0, 150.0, 130.0],
    }
    peak_memory = {
        15: [7100, 7400, 7200],
        20: [7500, 7600, 7400],
    }

    def fake_run(command, check):
        assert check is False
        commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        steps = int(command[command.index("--steps") + 1])
        repetition = int(output_dir.name.split("_")[-1])
        output_dir.mkdir(parents=True)
        output = output_dir / "render.mp4"
        preview = output_dir / "quality-samples.png"
        detail_preview = output_dir / "quality-details.png"
        for path in (output, preview, detail_preview):
            path.write_bytes(b"saved")
        (output_dir / "benchmark.json").write_text(
            json.dumps(
                {
                    "wall_seconds": wall_times[steps][repetition - 1],
                    "peak_memory_used_mb": peak_memory[steps][repetition - 1],
                    "quality_preview": str(preview),
                    "quality_detail_preview": str(detail_preview),
                    "output": str(output),
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("benchmark_full_body.subprocess.run", fake_run)
    args = Namespace(
        compare_steps=[15, 20],
        output_dir=tmp_path / "comparison",
        source_image=Path("target.png"),
        driving_video=Path("driver.mp4"),
        resolution=384,
        sample_seconds=0.25,
        fps=30.0,
        repetitions=3,
        visual_notes="",
    )

    assert _run_step_comparison(args) == 0
    report = json.loads(
        (args.output_dir / "quality-comparison.json").read_text(encoding="utf-8")
    )

    assert len(commands) == 6
    assert [run["steps"] for run in report["runs"]] == [15, 20, 15, 20, 15, 20]
    assert [run["repetition"] for run in report["runs"]] == [1, 1, 2, 2, 3, 3]
    assert all(
        command[command.index("--source-image") + 1] == "target.png"
        and command[command.index("--driving-video") + 1] == "driver.mp4"
        and command[command.index("--resolution") + 1] == "384"
        and command[command.index("--fps") + 1] == "30.0"
        for command in commands
    )
    assert [
        (
            setting["steps"],
            setting["successful_runs"],
            setting["median_wall_seconds"],
            setting["highest_peak_memory_used_mb"],
        )
        for setting in report["settings"]
    ] == [(15, 3, 120.0, 7400), (20, 3, 130.0, 7600)]
    assert len(report["visual_review"]["contact_sheets"]) == 6
    assert len(report["visual_review"]["detail_sheets"]) == 6
    assert all(Path(run["output"]).is_file() for run in report["runs"])
    assert len({run["run_dir"] for run in report["runs"]}) == 6
    assert all(
        (Path(run["run_dir"]) / "quality-samples.png").is_file()
        for run in report["runs"]
    )
    assert all(
        (Path(run["run_dir"]) / "quality-details.png").is_file()
        for run in report["runs"]
    )

