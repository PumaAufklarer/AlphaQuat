from alpha_quat.model.lightgbm.variants import VARIANTS
from alpha_quat.model.lightgbm.variants.quantile import QuantilePipeline


class TestQuantilePipelineVariant:
    def test_quantile_is_registered(self):
        assert "quantile" in VARIANTS
        assert VARIANTS["quantile"] is QuantilePipeline

    def test_quantile_pipeline(self):
        pipeline = QuantilePipeline()
        assert pipeline.mode == "quantile"
