"""Spoiled GRE simulation sub-routines."""

__all__ = ["SPGRModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch


class SPGRModel(AbstractModel):
    """SPGR transverse signal at time TE after excitation."""

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2star: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        field_map: float | npt.ArrayLike = 0.0,
        delta_cs: float | npt.ArrayLike = 0.0,
    ):
        self.properties.T1 = T1 * 1e-3
        self.properties.T2star = T2star * 1e-3
        self.properties.M0 = M0

        # We are assuming Freeman-Hill convention for off-resonance map,
        # so we need to negate to make use with this Ernst-Anderson-based implementation from Hoff
        self.properties.field_map = -2 * torch.pi * field_map
        self.properties.delta_cs = 2 * torch.pi * delta_cs

    @autocast
    def set_sequence(
        self,
        alpha: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike,
    ):
        self.sequence.alpha = torch.pi * alpha / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.TE = TE * 1e-3  # ms -> s

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
    ):
        # Prepare relaxation parameters
        R1, R2star = 1 / T1, 1 / T2star

        # Prepare off resonance
        df = field_map + delta_cs

        # Divide-by-zero risk with PyTorch's nan_to_num
        E1 = torch.exp(-R1 * TR)
        E2 = torch.exp(-R2star * TE)
        Phi = torch.exp(1j * df * TE)

        # Precompute cos, sin
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        # Main calculation
        den = 1 - E1 * ca
        Mxy = M0 * ((1 - E1) * sa) / den

        # Add decay
        signal = Mxy * E2

        # Add additional phase factor for readout at TE.
        signal = signal * Phi

        return signal
