import tqdm as tqdm
from torch.multiprocessing import Pool
from torch.optim import Adam
import torch as t
from evostrat import compute_centered_ranks, GaussianMixturePopulation
from evostrat.examples.lunar_lander import LunarLander

if __name__ == '__main__':
    """
    Lunar landers weights and biases are drawn from a mixture of 7 Gaussian distributions.    
    """
    param_shapes = {k: v.shape for k, v in LunarLander().get_params().items()}
    population = GaussianMixturePopulation(param_shapes, t.Size((7,)), LunarLander.from_params, std=0.1)

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
