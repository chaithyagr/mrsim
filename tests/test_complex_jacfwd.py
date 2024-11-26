"""
"""

import pytest
import torch

from mrsim.base.complex_jacfwd import complex_jacfwd


@complex_jacfwd(argnums=0)
def my_function(x: torch.Tensor) -> torch.Tensor:
    """
    Test function to validate complex_jacfwd behavior.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Complex-valued tensor.
    """
    return torch.exp(1j * x) * torch.sin(x)


@pytest.mark.parametrize(
    "input_tensor",
    [
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True),
        torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32, requires_grad=True),
    ],
)
def test_jacobian_output_shapes(input_tensor: torch.Tensor):
    """
    Test that the Jacobian computed by complex_jacfwd has the correct shape.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to test_function.
    """
    jacobian = my_function(input_tensor)

    # Ensure the Jacobian shape matches (N, N), where N is the size of the input tensor
    assert jacobian.shape == (input_tensor.size(0), input_tensor.size(0))
