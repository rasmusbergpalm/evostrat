# Torch EvoStrat 
A library that makes Evolutionary Strategies (ES) simple to use.

### Example

```python
pop = PopulationImpl(...)
optim = torch.optim.Adam(pop.parameters()) # Use any torch.optim optimizer
for i in range(N):
    optim.zero_grads()
    pop.fitness_grads(n_samples=200) # Computes approximate gradients
    optim.step()
```

For a complete example that solves 'LunarLander-v2' see [examples/lunar_lander.py](examples/lunar_lander.py)

![Lunar lander](media/lander.gif)

### Description

Evolutionary Strategies is a powerful approach to solve reinforcement learning problems and other optimization problems where the gradients cannot be computed with backprop. 
See ["Evolution strategies as a scalable alternative to reinforcement learning"](https://arxiv.org/abs/1703.03864) for an excellent introduction.

In ES the objective is to maximize the expected fitness of a distribution of individuals. 
With a few math tricks the gradient of this objective can be approximated, even if the fitness function itself is not differentiable, and maximized with gradient ascent. 

This library offers
 
1. A plug-and-play implementation of ES for pytorch reinforcement learning agents with `torch.nn.Module` policy networks. See [examples/lunar_lander.py](examples/lunar_lander.py) 
2. A simple and flexible interface for extending ES beyond the standard Normal distribution without having to derive any gradients by hand. Just subclass [Population](population.py)
 and implement the sampling process.    


     


 
