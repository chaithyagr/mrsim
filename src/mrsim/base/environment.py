"""Environmental parameters."""

__all__ = ["prepare_environmental_parameters"]

import math
import torch

from .decorators import broadcast

EPSILON = 1e-6  # To avoid division by zero or undefined values


def prepare_environmental_parameters(
    T2: torch.Tensor, T2_star: torch.Tensor = None, B0: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare environmental parameters for simulations.

    Parameters
    ----------
    T2 : torch.Tensor
        Transverse relaxation time (ms).
    T2_star : torch.Tensor, optional
        Effective transverse relaxation time including dephasing effects (ms).
        If not provided, R2' is set to 0.
    B0 : torch.Tensor, optional
        Off-resonance frequency (Hz). If not provided, it is set to 0.

    Returns
    -------
    R2 : torch.Tensor
        Transverse relaxation rate (1/s).
    R2_prime : torch.Tensor
        Rate of additional dephasing due to field inhomogeneities (1/s).
    B0_rad : torch.Tensor
        Off-resonance frequency (rad/s).

    Raises
    ------
    ValueError
        If T2_star is provided and T2_star >= T2.
    """
    # Handle T2*
    if T2_star is None:
        R2_prime = torch.zeros_like(T2)
    else:
        R2_prime = _prepare_R2_prime(T2, T2_star)

    # Handle B0 in Hz and convert to rad/s
    B0_rad = 2 * math.pi * B0 if B0 is not None else torch.zeros_like(T2)

    return _ret(R2_prime, B0_rad)


# %% local utils
@broadcast
def _prepare_R2_prime(T2, T2_star):
    if torch.any(T2_star >= T2):
        raise ValueError("T2* must be less than T2.")

    # Calculate R2 (1/s)
    R2 = 1.0 / (T2 + EPSILON)

    # Calculate R2' = R2* - R2 = (1/T2*) - R2 (1/s)
    R2_prime = 1.0 / (T2_star + EPSILON) - R2

    return R2_prime


@broadcast
def _ret(R2_prime, B0_rad):
    return R2_prime, B0_rad
