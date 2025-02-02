"""Balanced GRE simulation sub-routines."""

__all__ = ["bSSFPModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch


class bSSFPModel(AbstractModel):
    """
    bSSFPModel transverse signal at time TE after excitation.

    This class models the transverse magnetization signal generated by the
    balanced steady state free precession (bSSFP) sequence, calculated at echo time (TE)
    following RF excitation.

    Methods
    -------
    set_properties(T1, T2, M0=1.0, B0=0.0, chemshift=0.0):
        Set tissue and system-specific properties for the SPGR model.

    set_sequence(flip, TR, TE=None, phase_inc=180.0):
        Set sequence parameters including flip angle, repetition time (TR),
        echo time (TE) and RF phase increment.

    _engine(T1, T2, TR, TE, flip, M0=1.0, field_map=0.0, delta_cs=0.0):
        Compute the bSSFP signal for given tissue, sequence, and field parameters.

    Examples
    --------
    .. exec::

        from torchsim.models import bSSFPModel

        model = bSSFPModel()
        model.set_properties(T1=1000, T2=100)
        model.set_sequence(flip=60.0, TR=10.0, TE=5.0)
        signal = model()

    """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B0: float | npt.ArrayLike = 0.0,
        chemshift: float | npt.ArrayLike = 0.0,
    ):
        """
        Set tissue and system-specific properties for the SPGR model.

        Parameters
        ----------
        T1 : float | npt.ArrayLike
            Longitudinal relaxation time in milliseconds.
        T2 : float | npt.ArrayLike
            Transverse relaxation time in milliseconds.
        M0 : float | npt.ArrayLike, optional
            Proton density scaling factor, default is ``1.0``.
        B0 : float | npt.ArrayLike, optional
            Frequency offset map in Hz, default is ``0.0.``
        chemshift : float | npt.ArrayLik, optional
            Chemical shift in Hz, default is ``0.0``.

        """
        self.properties.T1 = T1
        self.properties.T2 = T2
        self.properties.M0 = M0
        self.properties.B0 = B0
        self.properties.chemshift = chemshift

    @autocast
    def set_sequence(
        self,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        phase_inc: float = 180.0,
        TE: float | npt.ArrayLike = None,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        flip : float | npt.ArrayLike
            Flip angle in degrees.
        TR : float | npt.ArrayLike
            Repetition time in milliseconds.
        phase_inc : float, optional
            Linear phase-cycle increment in degrees.
            The default is ``180.0``
        TE : float | npt.ArrayLike, optional
            Echo time in milliseconds.
            The default is ``None`` (i.e., ``TR/2``).

        """
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.phase_inc = torch.pi * phase_inc / 180.0
        if TE is None:
            TE = TR / 2
        self.sequence.TE = TE * 1e-3  # ms -> s

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B0: float | npt.ArrayLike = 0.0,
        chemshift: float | npt.ArrayLike = 0.0,
        phase_inc: float = 180.0,
    ):
        # Prepare relaxation parameters
        R1, R2 = 1e3 / T1, 1e3 / T2

        # We are assuming Freeman-Hill convention for off-resonance map,
        # so we need to negate to make use with this Ernst-Anderson-based implementation from Hoff
        B0 = -B0

        # Prepare off resonance
        df = 2 * torch.pi * (B0 + chemshift)

        # Divide-by-zero risk with PyTorch's nan_to_num
        E1 = torch.exp(-R1 * TR)
        E2 = torch.exp(-R2 * TE)
        Phi = torch.exp(1j * df * TE)

        # Precompute theta and some cos, sin
        theta = df * TR + phase_inc
        ca = torch.cos(flip)
        sa = torch.sin(flip)
        ct = torch.cos(theta)
        st = torch.sin(theta)

        # Main calculation
        den = (1 - E1 * ca) * (1 - E2 * ct) - (E2 * (E1 - ca)) * (E2 - ct)
        Mx = -1 * M0 * ((1 - E1) * E2 * sa * st) / den
        My = M0 * ((1 - E1) * sa) * (1 - E2 * ct) / den
        Mxy = Mx + 1j * My

        # Add decay
        signal = Mxy * E2

        # Add additional phase factor for readout at TE.
        signal = signal * Phi

        return signal
