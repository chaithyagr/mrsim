"""MPRAGE sub-routines."""

__all__ = ["MPRAGEModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch

from .. import epg


class MPRAGEModel(AbstractModel):
    """
    Magnetization Prepared RApid Gradient Echo (MPnRAGE) Model.

    This class models Magnetization Prepared RApid Gradient Echo (MPRAGE) signals
    based on tissue properties, pulse sequence parameters, and experimental conditions.
    It uses Extended Phase Graph (EPG) formalism to compute the magnetization evolution over time.

    Assume that signal is sampled at center of k-space only.

    Methods
    -------
    set_properties(T1, M0=1.0, inv_efficiency=1.0)
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(nshots, flip, TR, TI=0.0)
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, TI, flip, TRspgr, nshots, M0=1.0, inv_efficiency=1.0)
        Computes the MPRAGE signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        from torchsim.models import MPRAGEModel

        model = MPRAGEModel()
        model.set_properties(T1=(200, 1000), inv_efficiency=0.95)
        model.set_sequence(TI=500.0, flip=5.0, TRspgr=5.0, nshots=128)
        signal = model()

    """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ):
        """
        Set tissue and system-specific properties for the MRF model.

        Parameters
        ----------
        T1 : float | npt.ArrayLike
            Longitudinal relaxation time in milliseconds.
        M0 : float or array-like, optional
            Proton density scaling factor, default is ``1.0``.
        inv_efficiency : float | npt.ArrayLike, optional
            Inversion efficiency map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.M0 = M0
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        TI: float,
        flip: float,
        TRspgr: float,
        nshots: int | npt.ArrayLike,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        TI : float
            Inversion time in milliseconds of shape ``(2,)``.
        flip : float | npt.ArrayLike
            Flip angle train in degrees.
        TRspgr : float
            Repetition time in milliseconds for each SPGR readout.
        TRmprage : float
            Repetition time in milliseconds for the whole inversion block.
        nshots : int | npt.ArrayLike
            Number of SPGR readout within the inversion block of shape ``(npre, npost)``
            If scalar, assume ``npre == npost == 0.5 * nshots``. Usually, this
            is the number of slice encoding lines ``(nshots = nz / Rz)``,
            i.e., the number of slices divided by the total acceleration factor along ``z``.

        """
        self.sequence.nshots = nshots
        self.sequence.TI = TI * 1e-3  # ms -> s
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TRspgr = TRspgr * 1e-3  # ms -> s
        if nshots.numel() == 1:
            nshots = torch.repeat_interleave(nshots // 2, 2)
        self.sequence.nshots = nshots

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float,
        TRmprage: float,
        nshots: int | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ):
        R1 = 1e3 / T1

        # Calculate number of shots and time before DC sampling
        nshots_bef = nshots[0]
        time_bef = nshots_bef * TRspgr

        # Prepare EPG states matrix
        states = epg.states_matrix(
            device=R1.device,
            nstates=1,
        )
        # Prepare excitation pulse
        RF = epg.rf_pulse_op(flip)

        # Prepare relaxation operator for preparation pulse
        E1inv, rE1inv = epg.longitudinal_relaxation_op(R1, TI - time_bef)

        # Prepare relaxation operator for sequence loop
        E1, rE1 = epg.longitudinal_relaxation_op(R1, TRspgr)

        # Apply inversion
        states = epg.adiabatic_inversion(states, inv_efficiency)
        states = epg.longitudinal_relaxation(states, E1inv, rE1inv)
        states = epg.spoil(states)
        signal = []
        # Scan loop
        for p in range(nshots_bef):

            # Apply RF pulse
            states = epg.rf_pulse(states, RF)
            # Evolve
            states = epg.longitudinal_relaxation(states, E1, rE1)
            signal.append(M0 * 1j * epg.get_signal(states))
            states = epg.spoil(states)

        # Record signal
        return torch.stack(signal)
