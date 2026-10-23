import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils import driver_background as background


def test_composite_keeps_target_foreground_and_uses_driver_background():
    target = np.zeros((4, 4, 3), dtype=np.uint8)
    target[:] = (255, 0, 0)  # RGB red
    driver = np.zeros((4, 4, 3), dtype=np.uint8)
    driver[:] = (255, 0, 0)  # BGR blue
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[:, :2] = 1.0

    result = background.composite_target_over_driver_background(target, driver, alpha)

    assert tuple(result[1, 1]) == (255, 0, 0)
    assert tuple(result[1, 3]) == (0, 0, 255)


def test_driver_background_cover_crops_without_stretching():
    target = np.zeros((4, 4, 3), dtype=np.uint8)
    driver = np.zeros((4, 8, 3), dtype=np.uint8)
    driver[:, :4] = (0, 0, 255)  # left half in BGR
    driver[:, 4:] = (255, 0, 0)  # right half in BGR
    alpha = np.zeros((4, 4), dtype=np.float32)

    result = background.composite_target_over_driver_background(target, driver, alpha)

    assert result.shape == (4, 4, 3)
    # Center cropping retains the transition between the two halves.
    assert np.any(result[:, :, 0] == 255)
    assert np.any(result[:, :, 2] == 255)


def test_median_clean_background_ignores_a_transient_person():
    clean_room = np.full((4, 4, 3), 40, dtype=np.uint8)
    frames = [clean_room.copy() for _ in range(5)]
    frames[1][:, 1:3] = (0, 0, 255)
    frames[2][:, 1:3] = (0, 0, 255)

    result = background.median_clean_background(frames, (4, 4))

    assert np.array_equal(result, clean_room)


def test_extract_target_alpha_uses_isolated_worker_and_resizes(tmp_path, monkeypatch):
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"local test image")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\n"
        "from PIL import Image\n"
        "Image.new('RGBA', (2, 2), (100, 100, 100, 128)).save(sys.argv[2])\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(background, "BACKGROUND_REMOVAL_WORKER", worker)
    monkeypatch.setattr(background, "SF3D_ROOT", tmp_path)
    monkeypatch.setattr(background, "_photo_model_python", lambda: Path(sys.executable))

    alpha = background.extract_target_alpha(source, (6, 8))

    assert alpha.shape == (6, 8)
    assert np.all((alpha > 0.4) & (alpha < 0.6))


def test_extract_target_alpha_fails_clearly_without_local_model_environment(tmp_path, monkeypatch):
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"local test image")
    monkeypatch.setattr(
        background,
        "_photo_model_python",
        lambda: tmp_path / "missing-python",
    )

    try:
        background.extract_target_alpha(source, (4, 4))
    except background.DriverBackgroundError as exc:
        assert "Stable Fast 3D environment" in str(exc)
    else:
        raise AssertionError("Missing isolated model environment should fail explicitly.")
