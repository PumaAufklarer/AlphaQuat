from alpha_quat.model.lightgbm.variants import VARIANTS
from alpha_quat.model.lightgbm.variants.lambdarank import LambdaRankPipeline


class TestLambdaRankPipelineVariant:
    def test_lambdarank_is_registered(self):
        assert "lambdarank" in VARIANTS
        assert VARIANTS["lambdarank"] is LambdaRankPipeline

    def test_lambdarank_pipeline(self):
        pipeline = LambdaRankPipeline()
        assert pipeline.mode == "lambdarank"
