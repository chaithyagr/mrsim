"""Test RF operators."""

import pytest
import torch
from types import SimpleNamespace

from torchsim import epg


# Fixtures for common test inputs
@pytest.fixture
def states_fixture():
    return SimpleNamespace(
        Fplus=torch.tensor([0.0, 0.0, 0.0]),
        Fminus=torch.tensor([0.0, 0.0, 0.0]),
        Z=torch.tensor([1.0, 1.0, 1.0]),
    )


@pytest.fixture
def RF_fixture():
    T = torch.eye(3, dtype=torch.float32)
    RF = [
        [T[0][0][..., None], T[0][1][..., None], T[0][2][..., None]],
        [T[1][0][..., None], T[1][1][..., None], T[1][2][..., None]],
        [T[2][0][..., None], T[2][1][..., None], T[2][2][..., None]],
    ]
    return RF  # Identity RF operation


# Test functions
def test_rf_pulse_op():
    fa = torch.tensor(0.5)  # 0.5 radians
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)

    RF = epg.rf_pulse_op(fa, slice_prof, B1)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)


def test_phased_rf_pulse_op():
    fa = torch.tensor(0.5)
    phi = torch.tensor(0.2)
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)
    B1phase = torch.tensor(0.1)

    RF = epg.phased_rf_pulse_op(fa, phi, slice_prof, B1, B1phase)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)


def test_multidrive_rf_pulse_op():
    fa = torch.tensor([0.3, 0.4])
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor([0.8, 0.9])

    RF = epg.multidrive_rf_pulse_op(fa, slice_prof, B1)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)


def test_phased_multidrive_rf_pulse_op():
    fa = torch.tensor([0.3, 0.4])
    phi = torch.tensor([0.1, 0.2])
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor([0.8, 0.9])
    B1phase = torch.tensor([0.05, 0.07])

    RF, phi_out = epg.phased_multidrive_rf_pulse_op(fa, phi, slice_prof, B1, B1phase)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)
    assert torch.allclose(phi_out, torch.tensor([0.3]))


def test_initialize_mt_sat():
    duration = torch.tensor(0.001)  # 1 ms
    b1rms = torch.tensor(0.05)  # Tesla
    df = torch.tensor(0.0)
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)

    WT = epg.initialize_mt_sat(duration, b1rms, df, slice_prof, B1)

    assert isinstance(WT, torch.Tensor)


def test_mt_sat_op():
    WT = torch.tensor(-0.01)
    fa = torch.tensor(0.5)
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)

    exp_WT = epg.mt_sat_op(WT, fa, slice_prof, B1)

    assert isinstance(exp_WT, torch.Tensor)


def test_multidrive_mt_sat_op():
    WT = torch.tensor(-0.01)
    fa = torch.tensor([0.3, 0.4])
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor([0.8, 0.9])

    exp_WT = epg.multidrive_mt_sat_op(WT, fa, slice_prof, B1)

    assert isinstance(exp_WT, torch.Tensor)


def test_rf_pulse(states_fixture, RF_fixture):
    states = states_fixture
    RF = RF_fixture

    states_out = epg.rf_pulse(states, RF)

    assert torch.allclose(states_out.Fplus, states.Fplus)
    assert torch.allclose(states_out.Fminus, states.Fminus)
    assert torch.allclose(states_out.Z, states.Z)


def test_mt_sat(states_fixture):
    states = states_fixture
    Z = states.Z.clone()
    S = torch.tensor(0.9)

    states_out = epg.mt_sat(states, S)

    expected_Z = Z * S
    assert torch.allclose(states_out.Z, expected_Z)
