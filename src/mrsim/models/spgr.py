"""Spoiled GRE simulation sub-routines."""

__all__ = ["SPGRModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt


class SPGRModel(AbstractModel):
    """ """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2star: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        field_map: float | npt.ArrayLike = 0.0,
        delta_cs: float | npt.ArrayLike = 0.0,
    ): ...

    @autocast
    def set_sequence(
        self,
        alpha: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
    ): ...

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2star: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike,
        alpha: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        field_map: float | npt.ArrayLike = 0.0,
        delta_cs: float | npt.ArrayLike = 0.0,
    ): ...
