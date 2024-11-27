"""Generate EPG states matrix."""

__all__ = ["states_matrix"]

from types import SimpleNamespace

import torch


def states_matrix(
    dtype: torch.dtype,
    device: torch.device,
    nstates: int,
    nlocs: int = 1,
    ntrans_pools: int = 1,
    nlong_pools: int = 1,
) -> SimpleNamespace:
    """
    Generate EPG states matrix.

    Parameters
    ----------
    device : torch.device
        Computational device.
    nstates : int
        Numer of EPG states.
    nlocs : int, optional
        Number of spatial locations. The default is 1.
    ntrans_pools : int, optional
        Number of pools for transverse magnetization. The default is 1.
    nlong_pools : int, optional
        Number of pools for longitudinal magnetization. The default is 1.

    Returns
    -------
    states : SimpleNamespace
        EPG states matrix of with fields:

            * Fplus: transverse F+ states of shape (nstates, nlocs, ntrans_pools)
            * Fminus: transverse F- states of shape (nstates, nlocs, ntrans_pools)
            * Z: longitudinal Z states of shape (nstates, nlocs, nlong_pools)

    """
    Fplus = torch.zeros((nstates, nlocs, ntrans_pools), dtype=dtype, device=device)
    Fminus = torch.zeros((nstates, nlocs, ntrans_pools), dtype=dtype, device=device)
    Z = torch.zeros((nstates, nlocs, nlong_pools), dtype=dtype, device=device)
    Z[0] = 1.0

    return SimpleNamespace(Fplus=Fplus, Fminus=Fminus, Z=Z)
