"""BALM and network comparison models."""

__all__ = [
    "BALMHyperParams",
    "FitResult",
    "aligned_summary",
    "fit_balm",
    "predict",
    "template_draws",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import balm

    return getattr(balm, name)
