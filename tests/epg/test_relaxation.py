"""Test relaxation operators."""

import torch
from types import SimpleNamespace

from mrsim import epg


def test_longitudinal_relaxation_op():
    R1 = torch.tensor(0.5)
    time = torch.tensor(2.0)
    E1, rE1 = epg.longitudinal_relaxation_op(R1, time)

    # Expected results
    expected_E1 = torch.exp(-R1 * time)
    expected_rE1 = 1 - expected_E1

    assert torch.allclose(E1, expected_E1, atol=1e-6)
    assert torch.allclose(rE1, expected_rE1, atol=1e-6)

    # Test with batch inputs
    R1_batch = torch.tensor([0.5, 1.0])
    time_batch = torch.tensor([2.0, 3.0])
    E1_batch, rE1_batch = epg.longitudinal_relaxation_op(R1_batch, time_batch)

    expected_E1_batch = torch.exp(-R1_batch * time_batch)
    expected_rE1_batch = 1 - expected_E1_batch

    assert torch.allclose(E1_batch, expected_E1_batch, atol=1e-6)
    assert torch.allclose(rE1_batch, expected_rE1_batch, atol=1e-6)


def test_longitudinal_relaxation_exchange_op():
    weight = torch.tensor([0.5, 0.5])
    k = torch.tensor([[0, 0.2], [0.1, 0]])
    R1 = torch.tensor([0.5, 1.0])
    time = torch.tensor(2.0)
    E1, rE1 = epg.longitudinal_relaxation_exchange_op(weight, k, R1, time)

    # Shape validation
    assert E1.shape == k.shape
    assert rE1.shape == weight.shape

    # Numerical correctness can be tested with precomputed values or expected behavior
    assert E1.dtype == torch.complex64
    assert rE1.dtype == torch.complex64

    # Check with zero exchange rate
    k_zero = torch.zeros_like(k)
    E1_zero, rE1_zero = epg.longitudinal_relaxation_exchange_op(
        weight, k_zero, R1, time
    )
    expected_E1_zero, expected_rE1_zero = epg.longitudinal_relaxation_op(R1, time)

    assert torch.allclose(torch.diag(E1_zero.real), expected_E1_zero, atol=1e-6)
    assert torch.allclose(rE1_zero.real, weight * expected_rE1_zero, atol=1e-6)


def test_longitudinal_relaxation():
    Z = torch.tensor([1.0, 0.5])[None, :]
    states = SimpleNamespace(Z=Z.clone())
    E1 = torch.tensor(0.8)
    rE1 = torch.tensor(0.2)

    updated_states = epg.longitudinal_relaxation(states, E1, rE1)

    # Expected results
    expected_Z = Z.clone() * E1
    expected_Z[0] += rE1

    assert torch.allclose(updated_states.Z, expected_Z, atol=1e-6)


def test_longitudinal_relaxation_exchange():
    Z = torch.tensor([1.0, 0.5])[None, :]
    states = SimpleNamespace(Z=Z.clone())
    E1 = torch.tensor([[0.8, 0.1], [0.2, 0.9]])
    rE1 = torch.tensor([0.2, 0.1])

    updated_states = epg.longitudinal_relaxation_exchange(states, E1, rE1)

    # Expected results
    expected_Z = torch.einsum("...ij,...j->...i", E1, Z.clone())
    expected_Z[0] += rE1

    assert torch.allclose(updated_states.Z, expected_Z, atol=1e-6)


def test_edge_cases():
    # Test with zero R1
    R1_zero = torch.tensor(0.0)
    time = torch.tensor(1.0)
    E1, rE1 = epg.longitudinal_relaxation_op(R1_zero, time)

    assert torch.allclose(E1, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(rE1, torch.tensor(0.0), atol=1e-6)

    # Test with zero time
    R1 = torch.tensor(0.5)
    time_zero = torch.tensor(0.0)
    E1, rE1 = epg.longitudinal_relaxation_op(R1, time_zero)

    assert torch.allclose(E1, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(rE1, torch.tensor(0.0), atol=1e-6)

    # Test with large time (approaching steady-state)
    large_time = torch.tensor(1e6)
    E1, rE1 = epg.longitudinal_relaxation_op(R1, large_time)

    assert torch.allclose(E1, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(rE1, torch.tensor(1.0), atol=1e-6)
