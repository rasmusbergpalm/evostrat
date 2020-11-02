import torch as t
from typing import Iterable, Callable, Tuple
from torch.multiprocessing import Pool

from evostrat import Individual


class Population:
    """
    A parameterized distribution over individuals.

    Meant to be optimized with `torch.optim` optimizers, as follows:

    pop = PopulationImpl(...)
    optim = torch.optim.Adam(pop.parameters())
    for i in range(N):
        optim.zero_grads()
        pop.fitness_grads(n_samples=200)
        optim.step()
    """

    def parameters(self) -> Iterable[t.Tensor]:
        """
        :return: The parameters of this population distribution.
        """

        raise NotImplementedError

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        """
        Sample n individuals and compute their log probabilities. The log probability computation MUST be differentiable.

        :param n: How many individuals to sample
        :return: n individuals and their log probability of being sampled: [(ind_1, log_prob_1), ..., (ind_n, log_prob_n)]
        """
        raise NotImplementedError

    def fitness_grads(
            self,
            n_samples: int,
            pool: Pool = None,
            fitness_shaping_fn: Callable[[Iterable[float]], Iterable[float]] = lambda x: x
    ):
        """
        Computes the (approximate) gradients of the expected fitness of the population.

        Uses torch autodiff to compute the gradients. The Individual.fitness does NOT need to be differentiable,
        but the log probability computations in Population.sample MUST be.

        :param n_samples: How many individuals to sample to approximate the gradient
        :param pool: Optional process pool to use when computing the fitness of the sampled individuals.
        :param fitness_shaping_fn: Optional function to modify the fitness, e.g. normalization, etc. Input is a list of n raw fitness floats. Output must also be n floats.
        :return: A (n,) tensor containing the raw fitness (before fitness_shaping_fn) for the n individuals.
        """

        individuals, log_probs = zip(*self.sample(n_samples))
        assert all(lp.ndim == 0 and lp.isfinite() and lp.grad_fn is not None for lp in log_probs), "log_probs must be differentiable finite scalars"

        if pool is not None:
            raw_fitness = pool.map(_fitness_fn_no_grad, individuals)
        else:
            raw_fitness = list(map(_fitness_fn_no_grad, individuals))

        fitness = fitness_shaping_fn(raw_fitness)

        t.mean(t.stack([(-ind_fitness * log_prob) for log_prob, ind_fitness in zip(log_probs, fitness)])).backward()

        return t.tensor(raw_fitness)


def _fitness_fn_no_grad(ind: Individual):
    with t.no_grad():
        return ind.fitness()
