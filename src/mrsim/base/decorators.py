"""
"""

import inspect

from functools import wraps
from typing import Callable

import torch

from mrinufft._array_compat import _to_torch, _get_leading_argument, _get_device


def broadcast(func: Callable) -> Callable:
    """
    Force all inputs to be torch tensors of the same size on the same device.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _fill_kwargs(func, args, kwargs)

        # convert everything to torch
        args, kwargs = _to_torch(*args, **kwargs)

        # force everything to be at least 1d
        args = [torch.atleast_1d(arg) for arg in args]
        kwargs = {key: torch.atleast_1d(value) for key, value in kwargs.items()}

        # get device from first positional or keyworded argument
        leading_arg = _get_leading_argument(args, kwargs)

        # get array module from leading argument
        device = _get_device(leading_arg)

        # move everything to the leading argument device
        args = [arg.to(device) for arg in args]
        kwargs = {key: value.to(device) for key, value in kwargs.items()}

        # broadcast
        kwargs_args = list(kwargs.values())
        items = torch.broadcast_tensors(*args, *kwargs_args)
        items = list(items)

        # replace positional
        for n in range(len(args)):
            args[n] = items[0]
            items.pop(0)

        # replace kwargs
        kwargs = dict(zip(list(kwargs.keys()), items))

        # run function
        return func(*args, **kwargs)

    return wrapper


def _fill_kwargs(func, args, kwargs):
    """This automatically fills missing kwargs with default values."""
    signature = inspect.signature(func)

    # Get number of arguments
    n_args = len(args)

    # Create a dictionary of keyword arguments and their default values
    _kwargs = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            _kwargs[k] = v.default
        else:
            _kwargs[k] = None

    # Merge the default keyword arguments with the provided kwargs
    for k in kwargs.keys():
        _kwargs[k] = kwargs[k]

    # Replace args
    _keys = list(_kwargs.keys())[n_args:]
    _values = list(_kwargs.values())[n_args:]

    return args, dict(zip(_keys, _values))
