"""Unbalanced SSFP MR Fingerprinting sub-routines."""

__all__ = ["MRFModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch

from .. import epg


class MRFModel(AbstractModel):
    """SSFP Magnetic Resonance Fingerprinting."""

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ):
        self.properties.T1 = T1 * 1e-3
        self.properties.T2 = T2 * 1e-3
        self.properties.M0 = M0
        self.properties.B1 = B1
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        alpha: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike = 0.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
        nreps: int = 1,
    ):
        self.sequence.alpha = torch.pi * alpha / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.TI = TI * 1e-3  # ms -> s
        self.nstates = nstates

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        alpha: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike = 0.0,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
        nreps: int = 2,
    ):
        # Prepare relaxation parameters
        R1, R2 = 1 / T1, 1 / T2

        # Prepare EPG states matrix
        states = epg.states_matrix(
            device=R1.device,
            dtype=torch.float32,
            nlocs=len(slice_prof),
            nstates=nstates,
        )

        # Prepare relaxation operator for preparation pulse
        E1inv, rE1inv = epg.longitudinal_relaxation_op(R1, TI)

        # Prepare relaxation operator for sequence loop
        E1, rE1 = epg.longitudinal_relaxation_op(R1, TR)
        E2 = epg.transverse_relaxation_op(R2, TR)

        # Get number of shots
        nshots = len(alpha)

        for r in range(nreps):
            signal = []

            # Apply inversion
            states = epg.adiabatic_inversion(states, inv_efficiency)
            # states = -inv_efficiency * states
            states = epg.longitudinal_relaxation(states, E1inv, rE1inv)
            states = epg.spoil(states)

            # Scan loop
            for p in range(nshots):
                RF = epg.rf_pulse_op(B1, alpha[p], slice_prof)

                # Apply RF pulse
                states = epg.rf_pulse(states, RF)

                # Record signal
                signal.append(epg.get_signal(states))

                # Evolve
                states = epg.longitudinal_relaxation(states, E1, rE1)
                states = epg.transverse_relaxation(states, E2)
                states = epg.shift(states)

        return torch.cat(signal)
