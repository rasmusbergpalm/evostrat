# EvoStrat 

A library that makes Evolutionary Strategies (ES) simple to use.

### Installation

`pip install evostrat`

### Usage

```python
pop = PopulationImpl(...) # See complete examples for implementations. 
optim = torch.optim.Adam(pop.parameters()) # Use any torch.optim optimizer
for i in range(N):
    optim.zero_grads()
    pop.fitness_grads(n_samples=200) # Computes approximate gradients
    optim.step()
```

For complete examples that solves 'LunarLander-v2' see the [examples/](evostrat/examples). 

![Lunar lander](media/lander.gif)

### Description

Evolutionary Strategies is a powerful approach to solve reinforcement learning problems and other optimization problems where the gradients cannot be computed with backprop. 
See ["Evolution strategies as a scalable alternative to reinforcement learning"](https://arxiv.org/abs/1703.03864) for an excellent introduction.

In ES the objective is to maximize the expected fitness of a distribution over individuals, referred to as the population. 
With a few math tricks this objective can be maximized with gradient ascent, even if the fitness function itself is not differentiable. 

This library offers
 
1. A flexible and natural interface for ES that cleanly separates the environment, the reinforcement learning agent, the population distribution and the optimization.    
2. A plug-and-play approach for reinforcement learning agents with `torch.nn.Module` policy networks. See [examples/lunar_lander.py](evostrat/examples/lunar_lander.py) and [examples/normal_lunar_lander.py](evostrat/examples/normal_lunar_lander.py). 
3. Several population distributions and variants
    1. [Independent Normal](evostrat/normal_population.py). equivalent to OpenAI ES or PEPG depending on whether the standard deviation is fixed or learned. See [examples/normal_lunar_lander.py](evostrat/examples/normal_lunar_lander.py)
    2. [Multivariate Normal](evostrat/multivariate_normal_population.py) with a full covariance matrix. Similar to CMA-ES. See [examples/multivariate_normal_lunar_lander.py](evostrat/examples/multivariate_normal_lunar_lander.py)
    3. [Categorical](evostrat/categorical_population.py). For agents with categorical parameters, demonstrating the ability to handle non-normal distributions. See the [examples/binary_lunar_lander.py](evostrat/examples/binary_lunar_lander.py). 
4. A simple interface for creating your own populations, without having to derive any gradients! Just subclass [Population](evostrat/population.py) and implement the sampling process. See the built in populations for inspiration.

### Attribution

If you use this software in your academic work please cite

``` 
@misc{palm2020,
  author = {Palm, Rasmus Berg},
  title = {EvoStrat},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rasmusbergpalm/evostrat}}
}
```
     


 
