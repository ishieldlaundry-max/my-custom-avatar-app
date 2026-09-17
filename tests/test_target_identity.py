import numpy as np
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline


def test_strict_target_identity_overrides_render_controls():
    pipeline = FasterLivePortraitPipeline.__new__(FasterLivePortraitPipeline)
    pipeline.model_dict = {}
    pipeline.cfg = OmegaConf.create(
        {
            "infer_params": {
                "strict_target_identity": True,
                "flag_relative_motion": False,
                "flag_do_crop": False,
                "flag_pasteback": False,
                "flag_stitching": False,
                "animation_region": "all",
                "flag_color_match": False,
                "flag_crop_driving_video": False,
                "flag_landmark_scale_normalization": False,
                "expression_only": False,
                "stitching_blending_radius": 0.9,
            }
        }
    )

    pipeline._enforce_target_identity()

    assert pipeline.cfg.infer_params.flag_relative_motion is True
    assert pipeline.cfg.infer_params.flag_do_crop is True
    assert pipeline.cfg.infer_params.flag_pasteback is True
    assert pipeline.cfg.infer_params.flag_stitching is True
    assert pipeline.cfg.infer_params.animation_region == "exp"
    assert pipeline.cfg.infer_params.flag_color_match is True
    assert pipeline.cfg.infer_params.flag_crop_driving_video is True
    assert pipeline.cfg.infer_params.flag_landmark_scale_normalization is True
    assert pipeline.cfg.infer_params.expression_only is True
    assert pipeline.cfg.infer_params.stitching_blending_radius == 0.45


def test_relative_expression_is_scaled_and_bounded_to_target_landmarks():
    pipeline = FasterLivePortraitPipeline.__new__(FasterLivePortraitPipeline)
    pipeline.model_dict = {}
    pipeline.cfg = OmegaConf.create(
        {
            "infer_params": {
                "flag_landmark_scale_normalization": True,
                "landmark_scale_min_ratio": 0.5,
                "landmark_scale_max_ratio": 2.0,
                "relative_motion_max_magnitude": 0.25,
            }
        }
    )
    source = {
        "exp": np.zeros((1, 2, 3), dtype=np.float32),
        "kp": np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
    }
    driving_zero = {
        "exp": np.zeros((1, 2, 3), dtype=np.float32),
        "kp": np.array([[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=np.float32),
    }
    driving = {
        "exp": np.full((1, 2, 3), 10.0, dtype=np.float32),
        "kp": driving_zero["kp"].copy(),
    }

    mapped = pipeline._map_relative_expression(source, driving, driving_zero)
    displacement = np.linalg.norm(mapped - source["exp"], axis=-1)

    assert np.all(displacement <= 0.25 + 1e-6)