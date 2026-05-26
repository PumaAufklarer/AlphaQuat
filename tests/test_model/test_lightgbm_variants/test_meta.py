from alpha_quat.model.lightgbm.variants import VARIANTS
from alpha_quat.model.lightgbm.variants.meta import MetaPipeline


class TestMetaPipelineVariant:
    def test_meta_is_registered(self):
        assert "meta" in VARIANTS
        assert VARIANTS["meta"] is MetaPipeline

    def test_meta_pipeline(self):
        pipeline = MetaPipeline()
        assert pipeline.mode == "meta"
        assert hasattr(pipeline, "_train")
