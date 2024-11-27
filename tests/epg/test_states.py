"""EPG States creation."""

import pytest
import torch
from types import SimpleNamespace

from mrsim.epg import states_matrix


def test_states_matrix_default():
    dtype = torch.float32
    device = torch.device("cpu")
    nstates = 5

    result = states_matrix(dtype, device, nstates)

    assert isinstance(result, SimpleNamespace), "Result should be a SimpleNamespace"
    assert result.Fplus.shape == (nstates, 1, 1), "Fplus shape mismatch"
    assert result.Fminus.shape == (nstates, 1, 1), "Fminus shape mismatch"
    assert result.Z.shape == (nstates, 1, 1), "Z shape mismatch"

    assert torch.all(result.Fplus == 0), "Fplus should be initialized to zeros"
    assert torch.all(result.Fminus == 0), "Fminus should be initialized to zeros"
    assert torch.all(result.Z == 1), "Z should be initialized to ones"


def test_states_matrix_non_default():
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nstates = 10
    nlocs = 4
    ntrans_pools = 3
    nlong_pools = 2

    result = states_matrix(dtype, device, nstates, nlocs, ntrans_pools, nlong_pools)

    assert isinstance(result, SimpleNamespace), "Result should be a SimpleNamespace"
    assert result.Fplus.shape == (nstates, nlocs, ntrans_pools), "Fplus shape mismatch"
    assert result.Fminus.shape == (
        nstates,
        nlocs,
        ntrans_pools,
    ), "Fminus shape mismatch"
    assert result.Z.shape == (nstates, nlocs, nlong_pools), "Z shape mismatch"

    assert result.Fplus.dtype == dtype, "Fplus dtype mismatch"
    assert result.Fminus.dtype == dtype, "Fminus dtype mismatch"
    assert result.Z.dtype == dtype, "Z dtype mismatch"

    assert result.Fplus.device == device, "Fplus device mismatch"
    assert result.Fminus.device == device, "Fminus device mismatch"
    assert result.Z.device == device, "Z device mismatch"

    assert torch.all(result.Fplus == 0), "Fplus should be initialized to zeros"
    assert torch.all(result.Fminus == 0), "Fminus should be initialized to zeros"
    assert torch.all(result.Z == 1), "Z should be initialized to ones"


@pytest.mark.parametrize(
    "nstates,nlocs,ntrans_pools,nlong_pools",
    [
        (1, 1, 1, 1),
        (5, 2, 3, 4),
        (10, 10, 10, 10),
    ],
)
def test_states_matrix_parametrized(nstates, nlocs, ntrans_pools, nlong_pools):
    dtype = torch.float32
    device = torch.device("cpu")

    result = states_matrix(dtype, device, nstates, nlocs, ntrans_pools, nlong_pools)

    assert isinstance(result, SimpleNamespace), "Result should be a SimpleNamespace"
    assert result.Fplus.shape == (nstates, nlocs, ntrans_pools), "Fplus shape mismatch"
    assert result.Fminus.shape == (
        nstates,
        nlocs,
        ntrans_pools,
    ), "Fminus shape mismatch"
    assert result.Z.shape == (nstates, nlocs, nlong_pools), "Z shape mismatch"
