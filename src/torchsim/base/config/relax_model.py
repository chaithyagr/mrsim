"""Relaxation model for different systems."""

__all__ = [
    "prepare_single_pool",
    "prepare_two_pool_bm",
    "prepare_two_pool_mt",
    "prepare_three_pool",
]

import torch

EPSILON = 1e-6


def _particle_conservation(k: torch.Tensor) -> torch.Tensor:
    """
    Adjust diagonal of exchange matrix by imposing particle conservation.

    Parameters
    ----------
    k : torch.Tensor
        The exchange matrix (1/s).

    Returns
    -------
    torch.Tensor
        The adjusted exchange matrix ensuring particle conservation.
    """
    # Get the number of pools (assumed to be the last dimension of the tensor)
    npools = k.shape[-1]

    for n in range(npools):
        # Set the diagonal to zero
        k[..., n, n] = 0.0
        # Adjust diagonal to ensure conservation: sum of outgoing = sum of incoming
        k[..., n, n] = -k[..., n].sum(dim=-1)

    return k


def build_two_pool_exchange_matrix(
    weight: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """
    Build the exchange matrix for Bloch-McConnell or MT model, ensuring particle conservation.

    Parameters
    ----------
    weight : torch.Tensor
        Fractional weight of pool a and b.
    k : torch.Tensor
        Non-directional exchange rate(s) between pool a and b (1/s).

    Returns
    -------
    torch.Tensor
        The exchange matrix.

    """
    kab = k * weight[..., 1]  # kab: Exchange rate from pool a to pool b
    kba = k * weight[..., 0]  # kba: Exchange rate from pool b to pool a

    # Build rows
    row1 = torch.stack((-kab, kba), dim=-1)
    row2 = torch.stack((kab, -kba), dim=-1)

    # Stack rows to create matrix
    exchange_matrix = torch.stack((row1, row2), dim=-2)

    # Apply particle conservation
    exchange_matrix = _particle_conservation(exchange_matrix)

    return exchange_matrix


def build_three_pool_exchange_matrix(
    weight: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
) -> torch.Tensor:
    """
    Build the exchange matrix for Bloch-McConnell + MT model, ensuring particle conservation.

    Parameters
    ----------
    weight : torch.Tensor
        Fractional weight of pool a, b and c.
    k1 : torch.Tensor
        Non-directional exchange rate(s) between pool a and b (1/s).
    k2 : torch.Tensor
        Non-directional exchange rate(s) between pool b and c (1/s).

    Returns
    -------
    torch.Tensor
        The exchange matrix.

    """
    kab = k1 * weight[..., 1]  # kab: Exchange rate from pool a to pool b
    kba = k1 * weight[..., 0]  # kba: Exchange rate from pool b to pool a
    kbc = k2 * weight[..., 2]  # kbc: Exchange rate from pool b to pool c
    kcb = k2 * weight[..., 1]  # kbc: Exchange rate from pool c to pool b

    row1 = torch.stack(
        (-kab, kba, torch.zeros_like(k1)), dim=-1
    )  # Exchange from pool a to pool b
    row2 = torch.stack(
        (kab, -kba - kbc, kcb), dim=-1
    )  # Exchange from pool b to pool a, and from pool b to pool c
    row3 = torch.stack(
        (torch.zeros_like(k1), kbc, -kcb), dim=-1
    )  # Exchange from pool c to pool b

    # Stack exchange rates to create matrix
    exchange_matrix = torch.stack((row1, row2, row3), dim=-2)

    # Apply particle conservation
    exchange_matrix = _particle_conservation(exchange_matrix)

    return exchange_matrix


def prepare_single_pool(
    T1: torch.Tensor, T2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare simulation parameters for a single pool system.

    Parameters
    ----------
    T1 : torch.Tensor
        Longitudinal relaxation time for the pool (ms).
    T2 : torch.Tensor
        Transverse relaxation time for the pool (ms).

    Returns
    -------
    tuple of torch.Tensor
        - Longitudinal relaxation rate(s) (1/s).
        - Transverse relaxation rate(s) (1/s).
    """
    # Convert relaxation times to rates (1/s)
    R1 = 1.0 / (T1 + EPSILON)
    R2 = 1.0 / (T2 + EPSILON)

    return R1[..., None], R2[..., None]


def prepare_two_pool_bm(
    T1a: torch.Tensor,
    T1b: torch.Tensor,
    T2a: torch.Tensor,
    T2b: torch.Tensor,
    k: torch.Tensor,
    weight_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare simulation parameters for a two-pool Bloch-McConnell model.

    Parameters
    ----------
    T1a : torch.Tensor
        Longitudinal relaxation time for pool a (ms).
    T1b : torch.Tensor
        Longitudinal relaxation time for pool b (ms).
    T2a : torch.Tensor
        Transverse relaxation time for pool a (ms).
    T2b : torch.Tensor
        Transverse relaxation time for pool b (ms).
    k : torch.Tensor
        Exchange rate(s) between pool a and b (1/s).
    weight_b : torch.Tensor
        Fractional weight of pool b.

    Returns
    -------
    tuple of torch.Tensor
        - Longitudinal relaxation rate(s) for both pools (1/s).
        - Transverse relaxation rate(s) for both pools (1/s).
        - Exchange matrix.
        - Weight tensor for both pools.
    """
    # Convert relaxation times to rates (1/s)
    R1a = 1.0 / (T1a + EPSILON)
    R1b = 1.0 / (T1b + EPSILON)
    R2a = 1.0 / (T2a + EPSILON)
    R2b = 1.0 / (T2b + EPSILON)

    # Stack relaxation rates
    R1 = torch.stack((R1a, R1b), dim=-1)
    R2 = torch.stack((R2a, R2b), dim=-1)

    # Prepare weight tensor (pool a and pool b)
    weight_a = 1 - weight_b
    weight = torch.stack((weight_a, weight_b), dim=-1)

    # Build the exchange matrix using the two pool model
    exchange_matrix = build_two_pool_exchange_matrix(weight, k)

    return R1, R2, exchange_matrix, weight


def prepare_two_pool_mt(
    T1a: torch.Tensor,
    T2a: torch.Tensor,
    k: torch.Tensor,
    weight_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare simulation parameters for a two-pool Magnetization Transfer model.

    Parameters
    ----------
    T1a : torch.Tensor
        Longitudinal relaxation time for pool a (ms).
    T2a : torch.Tensor
        Transverse relaxation time for pool a (ms).
    k : torch.Tensor
        Exchange rate(s) between pool a and b (1/s).
    weight_b : torch.Tensor
        Fractional weight of pool b.

    Returns
    -------
    tuple of torch.Tensor
        - Longitudinal relaxation rate(s) for both pools (1/s).
        - Transverse relaxation rate(s) for both pools (1/s).
        - Exchange matrix.
        - Weight tensor for both pools.
    """
    # Convert relaxation times to rates (1/s)
    R1a = 1.0 / (T1a + EPSILON)
    R2a = 1.0 / (T2a + EPSILON)

    # Stack relaxation rates
    R1 = torch.stack((R1a, R1a), dim=-1)
    R2 = R2a[..., None]

    # Prepare weight tensor (pool a and pool b)
    weight_a = 1 - weight_b
    weight = torch.stack((weight_a, weight_b), dim=-1)

    # Build the exchange matrix using the two pool model
    exchange_matrix = build_two_pool_exchange_matrix(weight, k)

    return R1, R2, exchange_matrix, weight


def prepare_three_pool(
    T1a: torch.Tensor,
    T1b: torch.Tensor,
    T2a: torch.Tensor,
    T2b: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    weight_b: torch.Tensor,
    weight_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare simulation parameters for a three-pool Bloch-McConnell + Magnetization Transfer model.

    Parameters
    ----------
    T1a : torch.Tensor
        Longitudinal relaxation time for pool a (ms).
    T1b : torch.Tensor
        Longitudinal relaxation time for pool b (ms).
    T2a : torch.Tensor
        Transverse relaxation time for pool a (ms).
    k1 : torch.Tensor
        Exchange rate(s) between pool a and b (1/s).
    k2 : torch.Tensor
        Exchange rate(s) between pool b and c (1/s).
    weight_b : torch.Tensor
        Fractional weight of pool b.
    weight_c : torch.Tensor
        Fractional weight of pool c.

    Returns
    -------
    tuple of torch.Tensor
        - Longitudinal relaxation rate(s) for all pools (1/s).
        - Transverse relaxation rate(s) for all pools (1/s).
        - Exchange matrix.
        - Weight tensor for all pools.

    """
    # Convert relaxation times to rates (1/s)
    R1a = 1.0 / (T1a + EPSILON)
    R1b = 1.0 / (T1b + EPSILON)

    R2a = 1.0 / (T2a + EPSILON)
    R2b = 1.0 / (T2b + EPSILON)

    # Stack relaxation rates for all pools
    R1 = torch.stack((R1a, R1b, R1a), dim=-1)
    R2 = torch.stack((R2a, R2b), dim=-1)

    # Prepare weight tensor for all pools
    weight_a = 1 - weight_b - weight_c
    weight = torch.stack((weight_a, weight_b, weight_c), dim=-1)

    # Build the exchange matrix using the two pool model
    exchange_matrix = build_three_pool_exchange_matrix(weight, k1, k2)

    return R1, R2, exchange_matrix, weight
