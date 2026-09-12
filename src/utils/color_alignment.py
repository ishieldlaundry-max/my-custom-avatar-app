"""Color alignment helpers for synthesized portrait crops."""

import cv2
import numpy as np
import torch


def _as_rgb_uint8(image):
    """Return an HWC RGB uint8 array without modifying the input."""
    if torch.is_tensor(image):
        image = image.detach().to(dtype=torch.uint8).cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Color alignment expects an HWC RGB image.")
    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def align_lab_color(generated, reference, mask=None, strength=1.0):
    """Match generated LAB channel statistics to a reference portrait crop.

    A soft mask limits both the sampled pixels and the correction, which keeps
    background pixels and feathered crop boundaries stable.
    """
    generated_rgb = _as_rgb_uint8(generated)
    reference_rgb = _as_rgb_uint8(reference)
    if generated_rgb.shape != reference_rgb.shape:
        reference_rgb = cv2.resize(
            reference_rgb,
            (generated_rgb.shape[1], generated_rgb.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    if mask is None:
        weights = np.ones(generated_rgb.shape[:2], dtype=np.float32)
    else:
        weights = np.asarray(mask, dtype=np.float32)
        if weights.ndim == 3:
            weights = weights.mean(axis=2)
        if weights.shape != generated_rgb.shape[:2]:
            weights = cv2.resize(
                weights,
                (generated_rgb.shape[1], generated_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if weights.max(initial=0.0) > 1.0:
            weights /= 255.0
        weights = np.clip(weights, 0.0, 1.0)

    sample = weights > 0.15
    if sample.sum() < 16:
        return generated_rgb.copy()

    generated_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    matched_lab = generated_lab.copy()

    for channel in range(3):
        generated_values = generated_lab[..., channel][sample]
        reference_values = reference_lab[..., channel][sample]
        generated_std = max(float(generated_values.std()), 1.0)
        reference_std = max(float(reference_values.std()), 1.0)
        scale = np.clip(reference_std / generated_std, 0.65, 1.55)
        matched_lab[..., channel] = (
            (generated_lab[..., channel] - generated_values.mean()) * scale
            + reference_values.mean()
        )

    matched_rgb = cv2.cvtColor(
        np.clip(matched_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    ).astype(np.float32)
    blend = np.clip(weights * float(strength), 0.0, 1.0)[..., None]
    return np.clip(
        generated_rgb.astype(np.float32) * (1.0 - blend) + matched_rgb * blend,
        0,
        255,
    ).astype(np.uint8)


def align_lab_color_tensor(generated, reference, mask=None, strength=1.0):
    """Apply LAB alignment entirely on the generated tensor's device."""
    device = generated.device
    output_dtype = generated.dtype
    generated_rgb = generated.to(device=device, dtype=torch.float32) / 255.0
    reference_rgb = torch.as_tensor(reference, device=device, dtype=torch.float32) / 255.0
    if reference_rgb.shape[:2] != generated_rgb.shape[:2]:
        reference_rgb = torch.nn.functional.interpolate(
            reference_rgb.permute(2, 0, 1).unsqueeze(0),
            size=generated_rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    if mask is None:
        weights = torch.ones(generated_rgb.shape[:2], device=device)
    else:
        weights = torch.as_tensor(mask, device=device, dtype=torch.float32)
        if weights.ndim == 3:
            weights = weights.mean(dim=2)
        if weights.shape != generated_rgb.shape[:2]:
            weights = torch.nn.functional.interpolate(
                weights[None, None],
                size=generated_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        if weights.max() > 1.0:
            weights = weights / 255.0
        weights = weights.clamp(0.0, 1.0)

    sample_weights = torch.where(weights > 0.15, weights, torch.zeros_like(weights))
    weight_sum = sample_weights.sum()
    if weight_sum < 16:
        return generated.clone()

    def rgb_to_lab(rgb):
        linear = torch.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055).pow(2.4),
        )
        matrix = torch.tensor(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]],
            device=device,
        )
        xyz = linear @ matrix.T
        xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=device)
        delta = 6.0 / 29.0
        f_xyz = torch.where(
            xyz > delta ** 3,
            xyz.clamp_min(0).pow(1.0 / 3.0),
            xyz / (3.0 * delta ** 2) + 4.0 / 29.0,
        )
        return torch.stack(
            (
                116.0 * f_xyz[..., 1] - 16.0,
                500.0 * (f_xyz[..., 0] - f_xyz[..., 1]),
                200.0 * (f_xyz[..., 1] - f_xyz[..., 2]),
            ),
            dim=-1,
        )

    def lab_to_rgb(lab):
        fy = (lab[..., 0] + 16.0) / 116.0
        fx = fy + lab[..., 1] / 500.0
        fz = fy - lab[..., 2] / 200.0
        f_xyz = torch.stack((fx, fy, fz), dim=-1)
        delta = 6.0 / 29.0
        xyz = torch.where(
            f_xyz > delta,
            f_xyz.pow(3.0),
            3.0 * delta ** 2 * (f_xyz - 4.0 / 29.0),
        )
        xyz = xyz * torch.tensor([0.95047, 1.0, 1.08883], device=device)
        matrix = torch.tensor(
            [[3.2404542, -1.5371385, -0.4985314],
             [-0.9692660, 1.8760108, 0.0415560],
             [0.0556434, -0.2040259, 1.0572252]],
            device=device,
        )
        linear = xyz @ matrix.T
        return torch.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * linear.clamp_min(0).pow(1.0 / 2.4) - 0.055,
        ).clamp(0.0, 1.0)

    generated_lab = rgb_to_lab(generated_rgb)
    reference_lab = rgb_to_lab(reference_rgb)
    normalized_weights = sample_weights / weight_sum
    generated_mean = (generated_lab * normalized_weights[..., None]).sum(dim=(0, 1))
    reference_mean = (reference_lab * normalized_weights[..., None]).sum(dim=(0, 1))
    generated_var = (
        (generated_lab - generated_mean).pow(2) * normalized_weights[..., None]
    ).sum(dim=(0, 1))
    reference_var = (
        (reference_lab - reference_mean).pow(2) * normalized_weights[..., None]
    ).sum(dim=(0, 1))
    scale = (
        reference_var.clamp_min(1.0).sqrt()
        / generated_var.clamp_min(1.0).sqrt()
    ).clamp(0.65, 1.55)
    matched_rgb = lab_to_rgb(
        (generated_lab - generated_mean) * scale + reference_mean
    )
    blend = (weights * float(strength)).clamp(0.0, 1.0)[..., None]
    aligned = (generated_rgb * (1.0 - blend) + matched_rgb * blend) * 255.0
    return aligned.clamp(0.0, 255.0).to(dtype=output_dtype)