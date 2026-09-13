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