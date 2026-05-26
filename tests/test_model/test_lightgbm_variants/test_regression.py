from alpha_quat.model.lightgbm.variants import VARIANTS
from alpha_quat.model.lightgbm.variants.regression import RegressionPipeline


class TestRegressionPipelineVariant:
    def test_regression_is_registered(self):
        assert "regression" in VARIANTS
        assert VARIANTS["regression"] is RegressionPipeline

    def test_regression_pipeline(self):
        pipeline = RegressionPipeline()
        assert pipeline.mode == "regression"
