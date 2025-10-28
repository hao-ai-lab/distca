from typing import Dict, List, Optional

import torch

#### Tool functions for splitting and gathering args ####
def _split_tensor(x: Optional[torch.Tensor], num_splits: int):
    if x is None:
        return (None,) * num_splits
    return x.split(x.shape[0] // num_splits, dim=0)

def repack_args(args: List[List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args)
    return [
        [a[i] for a in args]
        for i in range(num_splits)
    ]

def _repack_dicts(args: Dict[str, List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args.values())
    return [
        {k: a[i] for k, a in args.items()}
        for i in range(num_splits)
    ]

def splits_all(tensors: List[torch.Tensor], num_splits: int):
    splits = [_split_tensor(t, num_splits) for t in tensors]
    return repack_args(splits, num_splits)

def split_all_dict(tensors: Dict[str, torch.Tensor], num_splits: int):
    splits = {k: _split_tensor(v, num_splits) for k, v in tensors.items()}
    return _repack_dicts(splits, num_splits)


def _split_tensor_with_num_tokens(x: Optional[torch.Tensor], num_tokens_list: List[int], key:str=None):
    """Split `x` along dim 0 into chunks specified by `num_tokens_list`.

    If `x` is None, return a tuple of Nones of the same length as `num_tokens_list`.
    Asserts that the sum of the requested sections equals `x.shape[0]`.
    """
    if x is None:
        return tuple(None for _ in num_tokens_list)

    assert isinstance(num_tokens_list, (list, tuple)) and all(isinstance(n, int) and n >= 0 for n in num_tokens_list), "num_tokens_list must be a list/tuple of non-negative integers"

    total = x.shape[0]
    assert sum(num_tokens_list) == total, f"Sum of {num_tokens_list = } ({sum(num_tokens_list) = }) must equal tensor's first-dim size ({total}), {x.shape = }, {key = }"

    # torch.split supports a list of section sizes; returns a tuple of tensors
    return x.split(num_tokens_list, dim=0)

def split_all_dict_with_num_tokens(
    tensors: Dict[str, torch.Tensor], 
    num_tokens_list: list[int],
):
    splits = {}
    for k, v in tensors.items():
        splits[k] = _split_tensor_with_num_tokens(v, num_tokens_list, key=k)
    
    return _repack_dicts(splits, len(num_tokens_list))


def gather_tensor(tensors: List[torch.Tensor], num_splits: int):
    assert len(tensors) == num_splits
    if any(t is None for t in tensors):
        assert all(t is None for t in tensors), "None tensors in gather_tensor"
        return None
    return torch.cat(tensors, dim=0)
####
