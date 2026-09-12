import numpy as np
import torch

from src.utils.color_alignment import align_lab_color, align_lab_color_tensor
from src.utils.crop import prepare_paste_back


def test_lab_alignment_moves_generated_color_toward_reference():
    generated = np.full((32, 32, 3), (180, 105, 90), dtype=np.uint8)
    reference = np.full((32, 32, 3), (105, 150, 190), dtype=np.uint8)

    aligned = align_lab_color(generated, reference)

    before = np.abs(generated.astype(float) - reference).mean()
    after = np.abs(aligned.astype(float) - reference).mean()
    assert after < before


def test_lab_alignment_respects_zero_mask():
    generated = np.full((16, 16, 3), 80, dtype=np.uint8)
    reference = np.full((16, 16, 3), 200, dtype=np.uint8)

    aligned = align_lab_color(
        generated,
        reference,
        mask=np.zeros((16, 16), dtype=np.float32),
    )

    np.testing.assert_array_equal(aligned, generated)


def test_tensor_alignment_preserves_device_dtype_and_shape():
    generated = torch.full((24, 24, 3), 80.0)
    reference = np.full((24, 24, 3), 180, dtype=np.uint8)

    aligned = align_lab_color_tensor(generated, reference)

    assert aligned.device == generated.device
    assert aligned.dtype == generated.dtype
    assert aligned.shape == generated.shape
    assert torch.mean(torch.abs(aligned - 180)) < torch.mean(
        torch.abs(generated - 180)
    )


def test_paste_back_blending_radius_softens_mask_edge():
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    identity = np.eye(3, dtype=np.float32)

    hard = prepare_paste_back(mask, identity, (64, 64), blending_radius=0.0)
    soft = prepare_paste_back(mask, identity, (64, 64), blending_radius=1.0)

    assert set(np.unique(hard[..., 0])) <= {0.0, 1.0}
    assert np.any((soft[..., 0] > 0.0) & (soft[..., 0] < 1.0))