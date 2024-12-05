"""Fast Spin Echo sub-routines."""

__all__ = ["FSEModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch

from .. import epg


class FSEModel(AbstractModel):
    """
    Fast Spin Echo (FSE) Model.

    This class simulates fast spin echo (FSE) signals based on
    tissue properties, pulse sequence parameters, and experimental conditions. It
    uses Extended Phase Graph (EPG) formalism to compute the magnetization evolution
    over time.

    Methods
    -------
    set_properties(T1, T2, M0=1.0, B1=1.0)
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(flip, ESP, phases=0.0, TR=1e6, exc_flip=90.0, exc_phase=90.0,
                 slice_prof=1.0, nstates=10)
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, T2, flip, ESP, phases, TR=1e6, exc_flip=90.0, exc_phase=90.0,
            M0=1.0, B1=1.0, slice_prof=1.0, nstates=10)
        Computes the FSE signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        import torch
        from mrsim.models import FSEModel

        model = FSEModel()
        model.set_properties(T1=1000, T2=80, M0=1.0, B1=1.0)
        model.set_sequence(flip=180.0 * torch.ones(128), ESP=2.0, TR=5000.0)
        signal = model()

    """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
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

        """
        self.properties.T1 = T1 * 1e-3
        self.properties.T2 = T2 * 1e-3
        self.properties.M0 = M0
        self.properties.B1 = B1

    @autocast
    def set_sequence(
        self,
        flip: float | npt.ArrayLike,
        ESP: float,
        phases: float | npt.ArrayLike = 0.0,
        TR: float | npt.ArrayLike = 1e6,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        flip : float | npt.ArrayLike
            Refocusing flip angle train in degrees.
        ESP : float
            Echo spacing in milliseconds.
        phases : float | npt.ArrayLike, optional
            Refocusing flip angle phases in degrees.
            The default is ``90.0``.
        TR : float | npt.ArrayLike, optional
            Repetition time in milliseconds.
            The default is ``1e6``.
        exc_flip : float, optional
            Excitation flip angle train in degrees.
            The default is ``90.0``.
        exc_phase : float, optional
            Excitation flip angle phase in degrees.
            The default is ``90.0``.
        slice_prof : float | npt.ArrayLike, optional
            Flip angle scaling along slice profile.
            The default is ``1.0``.
        nstates : int, optional
            Number of EPG states to be retained.
            The default is ``10``.

        """
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.ESP = ESP * 1e-3  # ms -> s
        if phases.numel() == 1:
            phases = phases * torch.ones_like(flip)
        self.sequence.phases = torch.pi * phases / 180.0
        self.sequence.exc_flip = torch.pi * exc_flip / 180.0
        self.sequence.exc_phase = torch.pi * exc_phase / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.slice_prof = slice_prof
        self.sequence.nstates = nstates

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
    ):
        # Prepare relaxation parameters
        R1, R2 = 1 / T1, 1 / T2

        # Prepare EPG states matrix
        states = epg.states_matrix(
            device=R1.device,
            nlocs=slice_prof.numel(),
            nstates=nstates,
        )

        # Prepare matrix for RF excitation
        RFexc = epg.phased_rf_pulse_op(exc_flip, exc_phase, slice_prof, B1)

        # Prepare relaxation operator for sequence loop
        E1, rE1 = epg.longitudinal_relaxation_op(R1, 0.5 * ESP)
        E2 = epg.transverse_relaxation_op(R2, 0.5 * ESP)

        # Get number of shots
        etl = len(flip)

        # Initialize signal
        signal = []

        # Apply excitation
        states = epg.rf_pulse(states, RFexc)

        # Scan loop
        for p in range(etl):
            # Pre refocusing
            states = epg.longitudinal_relaxation(states, E1, rE1)
            states = epg.transverse_relaxation(states, E2)
            states = epg.shift(states)

            # Refocus
            RF = epg.phased_rf_pulse_op(flip[p], phases[p], slice_prof, B1)
            states = epg.rf_pulse(states, RF)

            # Post refocusing
            states = epg.shift(states)
            states = epg.longitudinal_relaxation(states, E1, rE1)
            states = epg.transverse_relaxation(states, E2)

            # Record signal
            signal.append(epg.get_demodulated_signal(states, phases[p]))

        # Get signal
        signal = torch.stack(signal)  # (etl,)
        signal = signal[..., None]  # (etl, 1)

        # Get elapsed time and time left before next TR
        elapsed_time = ESP * etl
        dt = TR - elapsed_time

        # Calculate relaxation until TR
        ETR = torch.exp(-R1 * dt)  # (nTR,)

        # Apply modulation
        signal = M0 * signal * (1 - ETR) / (1 - ETR * signal)  # (etl, nTR)

        return signal.swapaxes(-1, -2).ravel()  # (nTR*etl,)
