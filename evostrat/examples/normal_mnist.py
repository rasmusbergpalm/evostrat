from typing import Dict

import torch as t
import tqdm as tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from evostrat import compute_centered_ranks, Individual, NormalPopulation
from evostrat.examples.iterable_wrapper import IterableWrapper


class MNIST(Individual):
    def __init__(self, batch):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        super().__init__()
        self.batch = batch
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.Tanh(),
            nn.Linear(128, 10), nn.Softmax(dim=1)
        )
        self.net.to(self.device)

    @staticmethod
    def from_params(batch, params: Dict[str, t.Tensor]) -> 'MNIST':
        mnist = MNIST(batch)
        mnist.net.load_state_dict(params)
        return mnist

    def fitness(self, render=False) -> float:
        input, target = self.batch
        output = self.net(input.to(self.device))
        acc = t.mean((t.argmax(output, dim=1) == target.to(self.device)).to(t.float32))
        return acc

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()


if __name__ == '__main__':
    batch_size = 1024
    train_loader = iter(DataLoader(IterableWrapper(datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())), batch_size=batch_size, pin_memory=True))
    test_loader = iter(DataLoader(IterableWrapper(datasets.MNIST('.', train=False, transform=transforms.ToTensor())), batch_size=batch_size, pin_memory=True))
    batch = None


    def constructor(params):
        return MNIST.from_params(batch, params)


    param_shapes = {k: v.shape for k, v in MNIST(None).get_params().items()}
    population = NormalPopulation(param_shapes, constructor, "shared")

    learning_rate = 1e-2
    iterations = 1000
    pop_size = 200

    optim = Adam(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations), position=0, leave=True)

    for _ in pbar:
        batch = next(train_loader)
        optim.zero_grad()
        raw_fit = population.fitness_grads(pop_size, None, compute_centered_ranks)
        optim.step()
        pbar.set_description("fit avg: %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
