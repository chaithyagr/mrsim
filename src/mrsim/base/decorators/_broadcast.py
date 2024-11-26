"""
"""

__all__ = ["broadcast_arguments"]

import torch


def broadcast_arguments(*args, **kwargs) -> tuple[list, dict]:
    """
    Force all inputs to be torch tensors of the same size.
    """
    items, kwitems, indexes, keys = _get_tensor_args_kwargs(*args, **kwargs)
    tmp = torch.broadcast_tensors(*items, *list(kwitems.values()))
    for n in range(len(items)):
        items[n] = tmp[0]
        tmp.pop(0)
    kwitems = dict(zip(kwitems.keys(), tmp))

    for idx in indexes:
        args[idx] = items[idx]
    for key in kwitems.keys():
        kwargs[key] = kwitems[key]

    return args, kwargs


# %% subroutines
def _get_tensor_args_kwargs(*args, **kwargs):
    items = []
    kwitems = {}
    indexes = []
    keys = []
    for n in range(len(args)):
        if isinstance(args[n], torch.Tensor):
            items.append(args[n])
            indexes.append(n)
    for key in kwargs.keys():
        if isinstance(kwargs[key], torch.Tensor):
            kwitems[key] = kwargs[key]
            keys.append(key)

    return items, kwitems, indexes, keys
