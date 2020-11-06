from typing import Iterable, Dict, Callable, Tuple

from evostrat import Population, Individual
import torch as t
import torch.distributions as d


class GaussianMixturePopulation(Population):
    def __init__(self,
                 mixing_shapes: Dict[str, t.Size],
                 component_shapes: t.Size,
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 std: float,
                 ):
        self.mixing_logits = {k: t.zeros(shape + (component_shapes[0],), requires_grad=True) for k, shape in mixing_shapes.items()}
        self.component_means = t.randn(component_shapes, requires_grad=True)
        self.std = std
        self.constructor = individual_constructor

    def parameters(self) -> Iterable[t.Tensor]:
        return list(self.mixing_logits.values()) + [self.component_means]

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        samples = []

        components = d.Normal(loc=self.component_means, scale=self.std)
        for i in range(n):
            log_p = 0.0
            params = {}
            for k, logits in self.mixing_logits.items():
                mix = d.Categorical(logits=logits)
                expanded = components.expand(mix.batch_shape + components.batch_shape)
                gmm = d.MixtureSameFamily(mix, d.Independent(expanded, self.component_means.ndim - 1))
                with t.no_grad():
                    sample = gmm.sample()
                params[k] = sample
                log_p += gmm.log_prob(sample).sum()

            samples.append((self.constructor(params), log_p))

        return samples
