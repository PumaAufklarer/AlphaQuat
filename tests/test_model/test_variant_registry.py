import pytest

from alpha_quat.model.variant_registry import VariantRegistry


class TestVariantRegistry:
    def test_registers_variant_by_mode(self):
        registry = VariantRegistry[object]("test")

        @registry.register
        class DemoVariant:
            mode = "demo"

        assert registry.names() == ["demo"]
        assert registry["demo"] is DemoVariant
        assert registry.as_dict() == {"demo": DemoVariant}

    def test_rejects_duplicate_mode(self):
        registry = VariantRegistry[object]("test")

        @registry.register
        class FirstVariant:
            mode = "demo"

        with pytest.raises(ValueError, match="Duplicate test variant mode: demo"):

            @registry.register
            class SecondVariant:
                mode = "demo"

    def test_rejects_empty_mode(self):
        registry = VariantRegistry[object]("test")

        with pytest.raises(ValueError, match="missing non-empty mode"):

            @registry.register
            class MissingModeVariant:
                mode = ""
