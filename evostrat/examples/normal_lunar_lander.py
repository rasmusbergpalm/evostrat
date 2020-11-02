from typing import Dict

import gym
import torch as t
import tqdm as tqdm
from torch import nn
from torch.multiprocessing import Pool
from torch.optim import Adam

from evostrat import compute_centered_ranks, Individual, NormalPopulation


class NormalLunarLander(Individual):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.Tanh(),
            nn.Linear(32, 4), nn.Softmax(dim=0)
        )

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'NormalLunarLander':
        agent = NormalLunarLander()
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        env = gym.make("LunarLander-v2")
        obs = env.reset()
        done = False
        r_tot = 0
        while not done:
            action = self.action(obs)
            obs, r, done, _ = env.step(action)
            r_tot += r
            if render:
                env.render()

        env.close()
        return r_tot

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()

    def action(self, obs):
        with t.no_grad():
            return t.argmax(self.net(t.tensor(obs, dtype=t.float32))).item()


if __name__ == '__main__':
    param_shapes = {k: v.shape for k, v in NormalLunarLander().get_params().items()}
    population = NormalPopulation(param_shapes, NormalLunarLander.from_params, std=0.1)

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
