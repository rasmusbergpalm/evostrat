from typing import Iterable, Dict, Callable

import torch as t
import torch.distributions as d

from evostrat import Individual, Population


class CategoricalPopulation(Population):
    """
    A distribution over individuals whose parameters are sampled from categorical distributions
    """

    def __init__(self,
                 individual_parameter_shapes: Dict[str, t.Size],
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual]
                 ):
        """
        A distribution over individuals whose parameters are sampled from categorical distributions

        The individuals are sampled by sampling their parameters from independent categorical distributions and then calling individual_constructor with the sampled parameters.

        :param individual_parameter_shapes: The shapes of the parameters of an individual. The last dimension denotes the number of classes of the categorical distribution.
        :param individual_constructor: A function that constructs an individual from parameters (with shapes equal to individual_parameter_shapes).
        """
        self.logits = {k: t.zeros(shape, requires_grad=True) for k, shape in individual_parameter_shapes.items()}
        self.constructor = individual_constructor

    def parameters(self) -> Iterable[t.Tensor]:
        return self.logits.values()

    def sample(self, n) -> Iterable[Individual]:
        samples = []
        for _ in range(n):
            classes = {k: d.Categorical(logits=logits).sample().detach() for k, logits in self.logits.items()}

            samples.append((
                self.constructor(classes),
                sum([d.Categorical(logits=logits).log_prob(classes[k]).sum() for k, logits in self.logits.items()])
            ))

        return samples
