VARIANTS: dict[str, type] = {}


def register(cls):
    VARIANTS[cls.mode] = cls
    return cls


from . import regression_signal, quantile_signal, lambdarank_signal, meta_signal
