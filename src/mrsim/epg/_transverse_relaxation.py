"""Transverse Relaxation operator."""

__all__ = ["transverse_relaxation_op", "transverse_relaxation"]

from types import SimpleNamespace

import torch


def transverse_relaxation_op(
    R2: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare transverse relaxation operator.

    Parameters
    ----------
    R2 : torch.Tensor
        Transverse relaxation rate in ``1/s``.
    time : torch.Tensor
        Time interval in ``s``.

    Returns
    -------
    E2 : torch.Tensor
        Transverse relaxation operator.

    """
    E2 = torch.exp(-R2 * time)

    return E2


def transverse_relaxation(
    states: SimpleNamespace,
    E2: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply transverse relaxation.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    E2 : torch.Tensor
        Transverse relaxation operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    Fplus = states.Fplus
    Fminus = states.Fminus

    # apply
    Fplus = Fplus.clone() * E2  # F+
    Fminus = Fminus.clone() * E2  # F-

    # prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus
    return states
