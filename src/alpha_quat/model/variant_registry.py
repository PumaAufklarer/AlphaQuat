from typing import Generic, TypeVar

T = TypeVar("T")


class VariantRegistry(Generic[T]):
    def __init__(self, label: str) -> None:
        self.label = label
        self._variants: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        mode = getattr(cls, "mode", "")
        if not mode:
            raise ValueError(f"{cls.__name__} missing non-empty mode")
        if mode in self._variants:
            raise ValueError(f"Duplicate {self.label} variant mode: {mode}")
        self._variants[mode] = cls
        return cls

    def names(self) -> list[str]:
        return list(self._variants)

    def as_dict(self) -> dict[str, type[T]]:
        return self._variants

    def __getitem__(self, mode: str) -> type[T]:
        return self._variants[mode]
