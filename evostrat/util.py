from typing import Iterable
import torch as t


def compute_centered_ranks(x: Iterable[float]) -> Iterable[float]:
    """
    Compute centered ranks, e.g. [-81.0, 11.0, -0.5] --> [-0.5, 0.5, 0.0]
    """
    x = t.tensor(x)
    assert x.ndim == 1
    ranks = t.zeros((len(x),), dtype=t.long)
    ranks[x.argsort()] = t.arange(len(x))
    ranks = ranks.to(t.float32)
    ranks = ranks / (len(x) - 1)
    ranks = ranks - 0.5
    return ranks.tolist()


def normalize(x: Iterable[float]) -> Iterable[float]:
    """
    Normalize a list of floats to have zero mean and variance 1
    """
    x = t.tensor(x)
    assert x.ndim == 1
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x.tolist()
