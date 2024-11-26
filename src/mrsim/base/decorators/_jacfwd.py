"""
"""

__all__ = ["jacfwd"]

from functools import wraps

from typing import Callable, Optional
import torch


def jacfwd(argnums: int = 0, recombine_output: bool = True) -> Callable:
    """
    Decorator to compute the Jacobian of a function with complex-valued outputs.

    Parameters
    ----------
    fn : Callable
        Function to compute the Jacobian for.
    argnums : int, optional
        The argument index to compute the Jacobian with respect to (default is 0).
    recombine_output : bool, optional
        If True, recombine real and imaginary components of the Jacobian into a complex tensor.
        If False, leave the Jacobian in split (real/imaginary) form.

    Returns
    -------
    Callable
        Decorated function that computes the Jacobian with appropriate handling of complex outputs.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Wrapper function for differentiation.

            Returns
            -------
            Tuple[torch.Tensor, Optional[torch.Tensor]]
                Original function output and its Jacobian.
            """

            # Define a wrapper function to evaluate real-imag split output
            def wrapped_fn(*wrapped_args):
                return split_real_imag(fn(*wrapped_args, **kwargs))

            # Compute the Jacobian using jacfwd
            jacobian = torch.func.jacfwd(wrapped_fn, argnums=argnums)(*args)

            # Optionally recombine output if needed
            if recombine_output:
                jacobian = combine_real_imag(jacobian)

            return jacobian

        return wrapper

    return decorator


def split_real_imag(tensor: torch.Tensor) -> torch.Tensor:
    """Split complex tensor into real and imaginary components."""
    if torch.is_complex(tensor):
        return torch.stack([tensor.real, tensor.imag], dim=0)
    else:
        return torch.stack([tensor, torch.zeros_like(tensor)], dim=0)


def combine_real_imag(split_tensor: torch.Tensor) -> torch.Tensor:
    """Combine split real and imaginary components into a complex tensor."""
    output = torch.complex(split_tensor[0], split_tensor[1])
    if torch.isreal(output).all().item():
        return output.real
    return output
