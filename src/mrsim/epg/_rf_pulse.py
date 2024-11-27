"""RF pulse operators."""

__all__ = ["rf_pulse_op", "phased_rf_pulse_op", "rf_pulse"]

from types import SimpleNamespace

import torch


def rf_pulse_op(
    fa: float,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float | torch.Tensor = 1.0,
) -> tuple[tuple[torch.Tensor]]:
    """
    Build RF rotation matrix.

    Parameters
    ----------
    fa : float
        Nominal flip angle in rad.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is 1.0.
    B1 : float | torch.Tensor, optional
        Flip angle scaling factor. The default is 1.0.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    """
    # apply B1 effect
    fa = B1 * fa

    # apply slice profile
    fa = slice_prof * fa

    # calculate operator
    T00 = torch.cos(fa / 2) ** 2
    T01 = torch.sin(fa / 2) ** 2
    T02 = torch.sin(fa)
    T10 = T01.conj()
    T11 = T00
    T12 = torch.sin(fa)
    T20 = -0.5 * 1j * torch.sin(fa)
    T21 = 0.5 * 1j * torch.sin(fa)
    T22 = torch.cos(fa)

    # build rows
    T0 = [T00[..., None], T01[..., None], T02[..., None]]
    T1 = [T10[..., None], T11[..., None], T12[..., None]]
    T2 = [T20[..., None], T21[..., None], T22[..., None]]

    # build matrix
    T = [T0, T1, T2]

    return T


def phased_rf_pulse_op(
    fa: float,
    phi: float,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float | torch.Tensor = 1.0,
    B1phase: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """
    Build RF rotation matrix along arbitrary axis.

    Parameters
    ----------
    fa : float
        Nominal flip angle in rad.
    phi : float
        RF phase.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is 1.0.
    B1 : float | torch.Tensor, optional
        Flip angle scaling factor. The default is 1.0.
    B1 : float | torch.Tensor, optional
        Transmit field phase. The default is 0.0.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    """
    # apply B1 effect
    fa = B1 * fa

    # apply slice profile
    fa = slice_prof * fa

    # calculate operator
    T00 = torch.cos(fa / 2) ** 2
    T01 = torch.exp(2 * 1j * phi) * (torch.sin(fa / 2)) ** 2
    T02 = -1j * torch.exp(1j * phi) * torch.sin(fa)
    T10 = T01.conj()
    T11 = T00
    T12 = 1j * torch.exp(-1j * phi) * torch.sin(fa)
    T20 = -0.5 * 1j * torch.exp(-1j * phi) * torch.sin(fa)
    T21 = 0.5 * 1j * torch.exp(1j * phi) * torch.sin(fa)
    T22 = torch.cos(fa)

    # build rows
    T0 = [T00[..., None], T01[..., None], T02[..., None]]
    T1 = [T10[..., None], T11[..., None], T12[..., None]]
    T2 = [T20[..., None], T21[..., None], T22[..., None]]

    # build matrix
    T = [T0, T1, T2]

    return T


def rf_pulse(
    states: SimpleNamespace,
    RF: tuple[tuple[torch.Tensor]],
) -> SimpleNamespace:
    """
    Apply RF rotation, mixing EPG states.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    RF : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    FplusIn = states.Fplus
    FminusIn = states.Fminus
    ZIn = states.Z

    # prepare out
    FplusOut = FplusIn.clone()
    FminusOut = FminusIn.clone()
    ZOut = ZIn.clone()

    # apply
    FplusOut = RF[0][0] * FplusIn + RF[0][1] * FminusIn + RF[0][2] * ZIn
    FminusOut = RF[1][0] * FplusIn + RF[1][1] * FminusIn + RF[1][2] * ZIn
    ZOut = RF[2][0] * FplusIn + RF[2][1] * FminusIn + RF[2][2] * ZIn

    # prepare for output
    states.Fplus = FplusOut
    states.Fminus = FminusOut
    states.Z = ZOut

    return states


def rf_pulse_and_saturation(): ...
