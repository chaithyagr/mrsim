"""Unbalanced SSFP MR Fingerprinting sub-routines."""

__all__ = ["MRFModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch

from .. import epg


class MRFModel(AbstractModel):
    """
    SSFP Magnetic Resonance Fingerprinting (MRF) Model.

    This class simulates steady-state free precession (SSFP) MRF signals based on
    tissue properties, pulse sequence parameters, and experimental conditions. It
    uses Extended Phase Graph (EPG) formalism to compute the magnetization evolution
    over time.

    Methods
    -------
    set_properties(T1, T2, M0=1.0, B1=1.0, inv_efficiency=1.0)
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(flip, TR, TI=0.0, slice_prof=1.0, nstates=10, nreps=1)
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, T2, flip, TR, TI=0.0, M0=1.0, B1=1.0, inv_efficiency=1.0,
            slice_prof=1.0, nstates=10, nreps=1)
        Computes the MRF signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        import torch
        from mrsim.models import MRFModel

        model = MRFModel()
        model.set_properties(T1=1000, T2=80, M0=1.0, B1=1.0, inv_efficiency=0.95)
        model.set_sequence(flip=torch.linspace(5.0, 60.0, 1000), TR=10.0, nstates=20, nreps=1)
        signal = model()

    """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ):
        """
        Set tissue and system-specific properties for the MRF model.

        Parameters
        ----------
        T1 : float | npt.ArrayLike
            Longitudinal relaxation time in milliseconds.
        T2 : float | npt.ArrayLike
            Transverse relaxation time in milliseconds.
        M0 : float or array-like, optional
            Proton density scaling factor, default is ``1.0``.
        B1 : float | npt.ArrayLike, optional
            Flip angle scaling map, default is ``1.0``.
        inv_efficiency : float | npt.ArrayLike, optional
            Inversion efficiency map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.T2 = T2
        self.properties.M0 = M0
        self.properties.B1 = B1
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float = 0.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
        nreps: int = 1,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        flip : float | npt.ArrayLike
            Flip angle train in degrees.
        TR : float | npt.ArrayLike
            Repetition time in milliseconds.
        TI : float, optional
            Inversion time in milliseconds.
            The default is ``0.0``.
        slice_prof : float | npt.ArrayLike, optional
            Flip angle scaling along slice profile.
            The default is ``1.0``.
        nstates : int, optional
            Number of EPG states to be retained.
            The default is ``10``.
        nreps : int, optional
            Number of simulation repetitions.
            The default is ``1``.

        """
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.TI = TI * 1e-3  # ms -> s
        self.sequence.slice_prof = slice_prof
        self.sequence.nstates = nstates
        self.sequence.nreps = nreps

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float = 0.0,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
        nreps: int = 1,
    ):
        # Prepare relaxation parameters
        R1, R2 = 1e3 / T1, 1e3 / T2

        # Prepare EPG states matrix
        states = epg.states_matrix(
            device=R1.device,
            nlocs=slice_prof.numel(),
            nstates=nstates,
        )

        # Prepare relaxation operator for preparation pulse
        E1inv, rE1inv = epg.longitudinal_relaxation_op(R1, TI)

        # Prepare relaxation operator for sequence loop
        E1, rE1 = epg.longitudinal_relaxation_op(R1, TR)
        E2 = epg.transverse_relaxation_op(R2, TR)

        # Get number of shots
        nshots = len(flip)

        for r in range(nreps):
            signal = []

            # Apply inversion
            states = epg.adiabatic_inversion(states, inv_efficiency)
            states = epg.longitudinal_relaxation(states, E1inv, rE1inv)
            states = epg.spoil(states)

            # Scan loop
            for p in range(nshots):
                RF = epg.rf_pulse_op(flip[p], slice_prof, B1)

                # Apply RF pulse
                states = epg.rf_pulse(states, RF)

                # Record signal
                signal.append(epg.get_signal(states))

                # Evolve
                states = epg.longitudinal_relaxation(states, E1, rE1)
                states = epg.transverse_relaxation(states, E2)
                states = epg.shift(states)

        return M0 * 1j * torch.stack(signal)
