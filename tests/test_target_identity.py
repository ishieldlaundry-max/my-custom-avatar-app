import numpy as np
from omegaconf import OmegaConf

import src.pipelines.faster_live_portrait_pipeline as pipeline_module
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


def test_realtime_flag_is_forwarded_once(monkeypatch):
    pipeline = FasterLivePortraitPipeline.__new__(FasterLivePortraitPipeline)
    pipeline.cfg = OmegaConf.create(
        {"infer_params": {"flag_crop_driving_video": False}}
    )
    landmarks = np.zeros((106, 2), dtype=np.float32)
    motion = np.zeros((1, 1), dtype=np.float32)
    keypoints = np.zeros((1, 21, 3), dtype=np.float32)

    class Predictor:
        def __init__(self, result):
            self.result = result

        def predict(self, *_args):
            return self.result

    pipeline.model_dict = {
        "face_analysis": Predictor([landmarks]),
        "landmark": Predictor(landmarks),
        "motion_extractor": Predictor(
            (motion, motion, motion, motion, keypoints, motion, keypoints)
        ),
    }
    pipeline.src_lmk_pre = None
    pipeline.R_d_0 = None
    pipeline.x_d_0_info = None
    pipeline._enforce_target_identity = lambda: None
    pipeline._get_target_canvas_tensor = lambda _img: np.zeros(
        (4, 4, 3), dtype=np.uint8
    )

    monkeypatch.setattr(
        pipeline_module,
        "calc_eye_close_ratio",
        lambda _landmarks: np.zeros((1, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        pipeline_module,
        "calc_lip_close_ratio",
        lambda _landmarks: np.zeros((1, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_rotation_matrix",
        lambda *_args: np.eye(3, dtype=np.float32)[None],
    )

    forwarded = {}

    def fake_run(
        src_info,
        x_d_i_info,
        x_d_0_info,
        R_d_i,
        R_d_0,
        realtime,
        input_eye_ratio,
        input_lip_ratio,
        canvas,
        **kwargs,
    ):
        forwarded["realtime"] = realtime
        forwarded["kwargs"] = kwargs
        return np.zeros((4, 4, 3), dtype=np.uint8), canvas

    pipeline._run = fake_run

    pipeline.run(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        [object()],
        first_frame=True,
        realtime=True,
    )

    assert forwarded["realtime"] is True
    assert "realtime" not in forwarded["kwargs"]