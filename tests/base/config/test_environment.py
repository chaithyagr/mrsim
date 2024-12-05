"""Test environmental parameters pre-processing."""

import torch
from torch.testing import assert_close

from torchsim.base import prepare_environmental_parameters

EPSILON = 1e-6


def test_prepare_environmental_parameters_with_all_inputs():
    T2 = torch.tensor([100.0, 120.0], dtype=torch.float32)  # ms
    T2_star = torch.tensor([80.0, 90.0], dtype=torch.float32)  # ms
    B0 = torch.tensor([100.0, -50.0], dtype=torch.float32)  # Hz

    R2_prime, B0_rad = prepare_environmental_parameters(T2, T2_star, B0)

    # Expected results
    expected_R2 = 1.0 / (T2 + EPSILON)
    expected_R2_prime = 1.0 / (T2_star + EPSILON) - expected_R2
    expected_B0_rad = 2 * torch.pi * B0

    # Assertions
    assert_close(R2_prime, expected_R2_prime)
    assert_close(B0_rad, expected_B0_rad)


def test_prepare_environmental_parameters_without_T2_star():
    T2 = torch.tensor([100.0, 120.0], dtype=torch.float32)  # ms
    B0 = torch.tensor([100.0, -50.0], dtype=torch.float32)  # Hz

    R2_prime, B0_rad = prepare_environmental_parameters(T2, B0=B0)

    # Expected results
    expected_R2_prime = torch.zeros_like(T2)
    expected_B0_rad = 2 * torch.pi * B0

    # Assertions
    assert_close(R2_prime, expected_R2_prime)
    assert_close(B0_rad, expected_B0_rad)


def test_prepare_environmental_parameters_without_B0():
    T2 = torch.tensor([100.0, 120.0], dtype=torch.float32)  # ms
    T2_star = torch.tensor([80.0, 90.0], dtype=torch.float32)  # ms

    R2_prime, B0_rad = prepare_environmental_parameters(T2, T2_star)

    # Expected results
    expected_R2 = 1.0 / (T2 + EPSILON)
    expected_R2_prime = 1.0 / (T2_star + EPSILON) - expected_R2
    expected_B0_rad = torch.zeros_like(T2)

    # Assertions
    assert_close(R2_prime, expected_R2_prime)
    assert_close(B0_rad, expected_B0_rad)


def test_prepare_environmental_parameters_without_T2_star_and_B0():
    T2 = torch.tensor([100.0, 120.0], dtype=torch.float32)  # ms

    R2_prime, B0_rad = prepare_environmental_parameters(T2)

    # Expected results
    expected_R2_prime = torch.zeros_like(T2)
    expected_B0_rad = torch.zeros_like(T2)

    # Assertions
    assert_close(R2_prime, expected_R2_prime)
    assert_close(B0_rad, expected_B0_rad)


# def test_prepare_environmental_parameters_invalid_T2_star():
#     T2 = torch.tensor([100.0, 120.0], dtype=torch.float32)  # ms
#     T2_star = torch.tensor([120.0, 130.0], dtype=torch.float32)  # ms (invalid)

#     with pytest.raises(ValueError, match="T2* must be less than T2."):
#         prepare_environmental_parameters(T2, T2_star)
