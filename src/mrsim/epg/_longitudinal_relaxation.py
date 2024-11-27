"""Longitudinal Relaxation operator."""

__all__ = ["longitudinal_relaxation_op", "longitudinal_relaxation"]

from types import SimpleNamespace

import torch


def longitudinal_relaxation_op(
    R1: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare longitudinal relaxation and recovery operators.

    Parameters
    ----------
    R1 : torch.Tensor
        Longitudinal relaxation rate in ``1/s``.
    time : torch.Tensor
        Time interval in ``s``.

    Returns
    -------
    E1 : torch.Tensor
        Longitudinal relaxation operator.
    rE1 : torch.Tensor
        Longitudinal recovery operator.

    """
    E1 = torch.exp(-R1 * time)
    rE1 = 1 - E1

    return E1, rE1


def longitudinal_relaxation(
    states: SimpleNamespace,
    E1: torch.Tensor,
    rE1: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply longitudinal relaxation and recovery.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    E1 : torch.Tensor
        Longitudinal relaxation operator.
    rE1 : torch.Tensor
        Longitudinal recovery operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    # parse
    Z = states.Z

    # apply
    Z = Z.clone() * E1  # decay
    Z[0] = Z[0].clone() + rE1  # regrowth

    # prepare for output
    states.Z = Z
    return states
