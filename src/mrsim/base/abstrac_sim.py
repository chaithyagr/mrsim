"""
"""

from abc import ABC, abstractmethod

import math
import torch


def autocast(): ...


class AbstractModel(ABC):
    """ """

    def __init__(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
        diff: str | tuple[str] | None = None,
        *args,
        **kwargs,
    ): ...

    @autocast
    def set_properties(self, *args, **kwargs): ...

    @autocast
    def set_sequence(self, *args, **kwargs): ...

    def forward(self, *args, **kwargs): ...

    def jacobian(self, *args, **kwargs): ...

    @staticmethod
    def _engine(): ...


# %% todo: move


def complex_jacfwd(): ...


argmap = {}


def SPGRModel(AbstractModel):

    def __init__(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
        diff: str | tuple[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(batch_size, device, diff)

        self._forward = self._engine
        self._jacobian = complex_jacfwd(argmap)(self._engine)

    @autocast
    def set_properties(self, T1, T2star, M0, field_map=None, delta_cs=None): ...

    @autocast
    def set_sequence(self, alpha, TR, TE): ...


# %% subroutines
def _get_spgr_phase(T2star, TE, field_map, delta_cs):
    """Additional SPGR phase factors."""
    # Enable broadcasting
    T2star = T2star.unsqueeze(-1)
    delta_cs = delta_cs.unsqueeze(-1)
    field_map = field_map.unsqueeze(-1)
    TE = TE.unsqueeze(0)

    # Compute total phase accrual
    phi = 2 * math.pi * (delta_cs + field_map) * TE

    # Compute signal dampening
    exp_term = torch.exp(-TE / T2star)
    exp_term = torch.nan_to_num(exp_term, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.exp(1j * phi) * exp_term


def _spgr_engine(
    T1,
    T2star,
    M0,
    field_map,
    delta_cs,
    TR,
    TE,
    alpha,
):
    # Unit conversion
    T1 = T1 * 1e-3  # ms -> s
    T2star = T2star * 1e-3  # ms -> s
    TE = TE * 1e-3  # ms -> s
    TR = TR * 1e-3  # ms -> s
    alpha = torch.deg2rad(alpha)

    # We are assuming Freeman-Hill convention for off-resonance map,
    # so we need to negate to make use with this Ernst-Anderson-based implementation from Hoff
    field_map = -1 * field_map

    # divide-by-zero risk with PyTorch's nan_to_num
    E1 = torch.exp(
        -1
        * torch.nan_to_num(
            TR.unsqueeze(0) / T1.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0
        )
    )

    # Precompute cos, sin
    ca = torch.cos(alpha).unsqueeze(0)
    sa = torch.sin(alpha).unsqueeze(0)

    # Main calculation
    den = 1 - E1 * ca
    Mxy = M0 * ((1 - E1) * sa) / den
    Mxy = torch.nan_to_num(Mxy, nan=0.0, posinf=0.0, neginf=0.0)

    # Add additional phase factor for readout at TE.
    signal = Mxy * _get_spgr_phase(T2star, TE, field_map, delta_cs)

    # Move multi-contrast in front
    signal = signal.unsqueeze(0)
    signal = signal.swapaxes(0, -1)

    return signal.squeeze().to(torch.complex64)
