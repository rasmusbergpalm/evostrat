import tqdm as tqdm
from torch.multiprocessing import Pool
from torch.optim import Adam

from evostrat import compute_centered_ranks, MultivariateNormalPopulation
from evostrat.examples.lunar_lander import LunarLander

if __name__ == '__main__':
    """
    Lunar landers weights and biases are drawn from a multivariate normal distribution with learned means and a learned covariance matrix.
    
    This is similar to CMA-ES [1].
    [1] - Hansen, Nikolaus, and Andreas Ostermeier. "Completely derandomized self-adaptation in evolution strategies." Evolutionary computation 9.2 (2001): 159-195.
    """
    param_shapes = {k: v.shape for k, v in LunarLander().get_params().items()}
    population = MultivariateNormalPopulation(param_shapes, LunarLander.from_params)

    learning_rate = 0.1
    iterations = 1000
    pop_size = 200

    optim = Adam(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    pool = Pool()

    for _ in pbar:
        optim.zero_grad()
        raw_fit = population.fitness_grads(pop_size, pool, compute_centered_ranks)
        optim.step()
        pbar.set_description("fit avg: %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
        if raw_fit.mean() > 200:
            print("Solved.")
            break

    pool.close()
