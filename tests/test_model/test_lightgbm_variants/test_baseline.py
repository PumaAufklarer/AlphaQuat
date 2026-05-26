import pytest

from alpha_quat.model.lightgbm.variants import VARIANTS, register
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline


class TestLightGBMBasePipeline:
    def test_base_pipeline_is_abstract(self):
        with pytest.raises(TypeError):
            LightGBMBasePipeline()

    def test_variants_registry_empty_initially(self):
        assert isinstance(VARIANTS, dict)

    def test_register_decorator(self):
        @register
        class TestVariant(LightGBMBasePipeline):
            mode = "test_mode"

            def _train(self, data, config):
                return {"test": "model"}

        assert "test_mode" in VARIANTS
        assert VARIANTS["test_mode"] is TestVariant
        assert TestVariant.mode == "test_mode"
