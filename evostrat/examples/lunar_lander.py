from typing import Dict

import gym
from torch import nn
import torch as t

from evostrat import Individual


class LunarLander(Individual):
    """
    A lunar lander controlled by a feedforward policy network
    """

    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.Tanh(),
            nn.Linear(32, 4), nn.Softmax(dim=0)
        )

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'LunarLander':
        agent = LunarLander()
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
