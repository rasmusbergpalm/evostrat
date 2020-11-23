from typing import Iterable, Tuple, Dict, Callable
import torch.distributions as d

import torch as t

from evostrat import Population, Individual


class MultivariateNormalPopulation(Population):
    """
    A distribution over individuals whose parameters are sampled from a multivariate normal with a full covariance matrix.
    """

    def __init__(self,
                 individual_parameter_shapes: Dict[str, t.Size],
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 device="cpu"
                 ):
        """
        A distribution over individuals whose parameters are sampled from a multivariate normal with a full covariance matrix.

        :param individual_parameter_shapes: The shapes of the parameters of an individual
        :param individual_constructor: Constructs individuals from parameters
        """
        n_params = sum([s.numel() for s in individual_parameter_shapes.values()])
        self.means = t.zeros((n_params,), dtype=t.float32, requires_grad=True, device=device)
        self.log_stds = (-6 * t.ones((n_params, n_params)) + 6 * t.eye(n_params, n_params)).clone().detach().to(device).requires_grad_(True)
        self.shapes = individual_parameter_shapes
        self.constructor = individual_constructor

    def _to_shapes(self, flat_params) -> Dict[str, t.Tensor]:
        params = {}
        i = 0
        for k, s in self.shapes.items():
            params[k] = flat_params[i:i + s.numel()].reshape(s)
            i += s.numel()
        return params

    def parameters(self) -> Iterable[t.Tensor]:
        return [self.means, self.log_stds]

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        for _ in range(n):
            dist = d.MultivariateNormal(loc=self.means, scale_tril=t.exp(self.log_stds).tril())
            with t.no_grad():
                sample = dist.sample()
            log_prob = dist.log_prob(sample)
            yield self.constructor(self._to_shapes(sample)), log_prob
