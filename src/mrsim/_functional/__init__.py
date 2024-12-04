"""Functional interface for signal models."""

__all__ = []

from ._spgr import spgr_sim  # noqa

__all__.append("spgr_sim")


from ._mrf import mrf_sim  # noqa

__all__.append("mrf_sim")
