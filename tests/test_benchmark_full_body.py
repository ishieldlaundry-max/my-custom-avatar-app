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
        (output_dir / "benchmark.json").write_text(
            json.dumps(
                {
                    "steps": steps,
                    "wall_seconds": 120 + steps,
                    "peak_memory_used_mb": 7000 + steps,
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

