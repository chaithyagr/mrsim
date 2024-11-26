"""Test relaxation model pre-processing"""

import torch
from torch.testing import assert_close

from mrsim.base.relax_model import build_two_pool_exchange_matrix
from mrsim.base.relax_model import build_three_pool_exchange_matrix
from mrsim.base.relax_model import prepare_single_pool
from mrsim.base.relax_model import prepare_two_pool_bm
from mrsim.base.relax_model import prepare_two_pool_mt
from mrsim.base.relax_model import prepare_three_pool

EPSILON = 1e-6


# Test for build_exchange_matrix_bm
def test_build_two_pool_exchange_matrix():
    weight_b = torch.tensor([0.2, 0.4], dtype=torch.float32)
    k = torch.tensor([0.05, 0.15], dtype=torch.float32)

    weight = torch.stack((1 - weight_b, weight_b), dim=-1)
    exchange_matrix = build_two_pool_exchange_matrix(weight, k)

    # Expected shape
    assert exchange_matrix.shape == (2, 2, 2)


# Test consistency with Malik
def test_directional_exchange():
    k = build_two_pool_exchange_matrix(torch.tensor([0.8, 0.2]), 10.0)
    ka = -k[0, 0]
    tau_b = 1000.0 / k[0, 1]  # 1 / kb (ms)

    assert ka == 2
    assert tau_b == 125.0


# Test for build_exchange_matrix_bm_mt
def test_build_three_pool_exchange_matrix():
    weight_b = torch.tensor([0.2, 0.4], dtype=torch.float32)
    weight_c = torch.tensor([0.1, 0.2], dtype=torch.float32)
    k1 = torch.tensor([0.02, 0.06], dtype=torch.float32)
    k2 = torch.tensor([0.03, 0.08], dtype=torch.float32)

    weight = torch.stack((1 - weight_b - weight_c, weight_b, weight_c), dim=-1)
    exchange_matrix = build_three_pool_exchange_matrix(weight, k1, k2)

    # Expected shape
    assert exchange_matrix.shape == (2, 3, 3)


# Test for prepare_single_pool
def test_prepare_single_pool():
    T1 = torch.tensor([800, 1200], dtype=torch.float32)
    T2 = torch.tensor([100, 120], dtype=torch.float32)

    R1, R2 = prepare_single_pool(T1, T2)

    # Check relaxation rates
    assert_close(R1[..., 0], 1.0 / (T1 + EPSILON))
    assert_close(R2[..., 0], 1.0 / (T2 + EPSILON))


# Test for prepare_two_pool_bm
def test_prepare_two_pool_bm():
    T1a = torch.tensor([800, 1200], dtype=torch.float32)
    T1b = torch.tensor([700, 1100], dtype=torch.float32)
    T2a = torch.tensor([80, 90], dtype=torch.float32)
    T2b = torch.tensor([90, 110], dtype=torch.float32)
    weight_b = torch.tensor([0.4, 0.6], dtype=torch.float32)
    k = torch.tensor([0.02, 0.06], dtype=torch.float32)

    R1, R2, exchange_matrix, weight = prepare_two_pool_bm(
        T1a, T1b, T2a, T2b, weight_b, k
    )

    # Check relaxation rates
    assert_close(R1[..., 0], 1.0 / (T1a + EPSILON))
    assert_close(R1[..., 1], 1.0 / (T1b + EPSILON))
    assert_close(R2[..., 0], 1.0 / (T2a + EPSILON))
    assert_close(R2[..., 1], 1.0 / (T2b + EPSILON))

    # Check exchange matrix shape
    assert exchange_matrix.shape == (2, 2, 2)


# Test for prepare_two_pool_mt
def test_prepare_two_pool_mt():
    T1a = torch.tensor([800, 1200], dtype=torch.float32)
    T2a = torch.tensor([100, 120], dtype=torch.float32)
    weight_b = torch.tensor([0.3, 0.5], dtype=torch.float32)
    k = torch.tensor([0.05, 0.1], dtype=torch.float32)

    R1, R2, exchange_matrix, weight = prepare_two_pool_mt(T1a, T2a, weight_b, k)

    # Check relaxation rates
    assert_close(R1[..., 0], 1.0 / (T1a + EPSILON))
    assert_close(R1[..., 1], 1.0 / (T1a + EPSILON))
    assert_close(R2[..., 0], 1.0 / (T2a + EPSILON))

    # Check exchange matrix shape
    assert exchange_matrix.shape == (2, 2, 2)


# Test for prepare_three_pool
def test_prepare_three_pool():
    T1a = torch.tensor([800, 1200], dtype=torch.float32)
    T1b = torch.tensor([700, 1100], dtype=torch.float32)
    T2a = torch.tensor([100, 120], dtype=torch.float32)
    T2b = torch.tensor([90, 110], dtype=torch.float32)
    weight_b = torch.tensor([0.3, 0.5], dtype=torch.float32)
    weight_c = torch.tensor([0.2, 0.4], dtype=torch.float32)
    k1 = torch.tensor([0.05, 0.1], dtype=torch.float32)
    k2 = torch.tensor([0.07, 0.12], dtype=torch.float32)

    R1, R2, exchange_matrix, weight = prepare_three_pool(
        T1a, T1b, T2a, T2b, k1, k2, weight_b, weight_c
    )

    # Check relaxation rates
    assert_close(R1[..., 0], 1.0 / (T1a + EPSILON))
    assert_close(R1[..., 1], 1.0 / (T1b + EPSILON))
    assert_close(R1[..., 2], 1.0 / (T1a + EPSILON))
    assert_close(R2[..., 0], 1.0 / (T2a + EPSILON))
    assert_close(R2[..., 1], 1.0 / (T2b + EPSILON))

    # Check exchange matrix shape
    assert exchange_matrix.shape == (2, 3, 3)
