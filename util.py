from typing import Iterable

import numpy as np


def compute_centered_ranks(x: Iterable[float]) -> Iterable[float]:
    def compute_ranks(x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    x = np.array(x)
    assert x.ndim == 1
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y.tolist()
