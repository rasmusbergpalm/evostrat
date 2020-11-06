from typing import Iterable, Dict, Callable, Tuple

from evostrat import Population, Individual
import torch as t
import torch.distributions as d


class GaussianMixturePopulation(Population):
    """
    A distribution over individuals whose parameters are sampled from a Gaussian Mixture Model.
    """

    def __init__(self,
                 mixing_shapes: Dict[str, t.Size],
                 component_shapes: t.Size,
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 std: float,
                 ):
        """
        A distribution over individuals whose parameters are sampled from a Gaussian Mixture Model.

        The shape of the parameters that individual_constructor is called with is mixing_shapes.shape + component_shapes[1:]

        Examples:
            A (3, 5) mixing shape and a (7,) component shape will result in the constructor being called with (3, 5) independent samples from a mixture of 7 (1-D) Gaussian distributions.
            A (4, 6) mixing shape and a (7, 2) component shape will result in the constructor being called with a (4, 6, 2) tensor corresponding to (4, 6) independent samples from a mixture of 7 2-D Gaussian distributions.

        :param mixing_shapes: The shapes of the parameters that are sampled from a Mixture of Gaussians.
        :param component_shapes: The shapes of the Gaussian components. The first dimension is the number of components. Remaining dimensions are the shape of the Gaussian distributions.
        Examples:
            - t.Size((7,)) means the parameters will be a mixture of 7 (1-D) Gaussian distributions.
            - t.Size((7, 2)) means the parameters will be a mixture of 7 2-D Gaussian distributions.
        :param individual_constructor: A function that constructs an individual from parameters with shapes mixing_shapes.shape + component_shapes[1:]
        :param std: The fixed std deviation of all the Gaussians
        """
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
