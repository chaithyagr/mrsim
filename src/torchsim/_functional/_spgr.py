"""SPoiled Gradient Recalled echo simulator."""

__all__ = ["spgr_sim"]

import numpy.typing as npt
import torch

from ..models.spgr import SPGRModel


def spgr_sim(
    flip: float | npt.ArrayLike,
    TE: float | npt.ArrayLike,
    TR: float,
    T1: float | npt.ArrayLike,
    T2star: float | npt.ArrayLike,
    diff: str | tuple[str] = None,
    B0: float | npt.ArrayLike = 0.0,
    chemshift: float | npt.ArrayLike = 0.0,
    M0: float | npt.ArrayLike = 1.0,
    chunk_size: int = None,
    device: str | torch.device = None,
):
    """
    SPoiled Gradient Recalled echo simulator wrapper.

    Parameters
    ----------
    flip : float | npt.ArrayLike
        Flip angle train in degrees.
    TE : float | npt.ArrayLike
        Echo time in milliseconds.
    TR : float | npt.ArrayLike
        Repetition time in milliseconds.
    T1 : float | npt.ArrayLike
        Longitudinal relaxation time in milliseconds.
    T2star : float | npt.ArrayLike
        Effective transverse relaxation time in milliseconds.
    diff : str | tuple[str], optional
        Arguments to get the signal derivative with respect to.
        The default is ``None`` (no differentation).
    B0 : float | npt.ArrayLike, optional
        Frequency offset map in Hz, default is ``0.0.``
    chemshift : float | npt.ArrayLik, optional
        Chemical shift in Hz, default is ``0.0``.
    M0 : float or array-like, optional
        Proton density scaling factor, default is ``1.0``.
    chunk_size : int, optional
        Number of atoms to be simulated in parallel.
        The default is ``None``.
    device : str | torch.device, optional
        Computational device for simulation.
        The default is ``None`` (infer from input).

    Returns
    -------
    sig : npt.ArrayLike
        Signal evolution of shape ``(..., len(flip))``.
    jac : npt.ArrayLike
        Derivatives of signal wrt ``diff`` parameters,
        of shape ``(..., len(diff), len(flip))``.
        Not returned if ``diff`` is ``None``.

    """
    model = SPGRModel(diff, chunk_size, device)
    model.set_properties(T1, T2star, M0, B0, chemshift)
    model.set_sequence(flip, TR, TE)
    return model()
