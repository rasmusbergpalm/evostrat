import unittest
import torch as t

from evostrat.gmm_population import GaussianMixturePopulation
from torch.distributions import Normal


class TestGMMPopulation(unittest.TestCase):

    def test_logp_grad(self):
        std = 0.1
        gmp = GaussianMixturePopulation({'a': t.Size((1,))}, t.Size((2,)), lambda x: x, std)
        gmp.mixing_logits = {'a': t.tensor([1.0, 0.0], requires_grad=True)}
        gmp.component_means = t.tensor([1.0, 3.0], requires_grad=True)
        ind, logp = next(gmp.sample(1))
        x = ind['a'].detach()

        p_z = t.softmax(gmp.mixing_logits['a'], dim=0)
        log_p_x = t.log(Normal(gmp.component_means[0], std).log_prob(x).exp() * p_z[0] +
                        Normal(gmp.component_means[1], std).log_prob(x).exp() * p_z[1])

        params = gmp.parameters()
        e_grads = t.autograd.grad(log_p_x, params)
        a_grads = t.autograd.grad(logp, params)

        for e, a in zip(e_grads, a_grads):
            self.assertTrue(t.allclose(e, a))

    def test_logp(self):
        gmp = GaussianMixturePopulation({'a': t.Size((1,))}, t.Size((2,)), lambda x: x, 1.0)
        gmp.mixing_logits = {'a': t.tensor([1.0, 0.0])}
        gmp.component_means = t.tensor([1.0, 3.0])
        ind, logp = next(gmp.sample(1))
        x = ind['a']

        p_z = t.softmax(gmp.mixing_logits['a'], dim=0)
        p_x = (Normal(gmp.component_means[0], 1.0).log_prob(x).exp() * p_z[0] +
               Normal(gmp.component_means[1], 1.0).log_prob(x).exp() * p_z[1])

        self.assertTrue(t.allclose(p_x, logp.exp()))

    def test_gmm(self):
        t.distributions.Distribution.set_default_validate_args(False)
        gmp = GaussianMixturePopulation({'a': t.Size((30, 50))}, t.Size((7,)), lambda x: x, 0.0)

        inds, logps = zip(*gmp.sample(9))

        self.assertEqual(len(inds), 9)
        self.assertEqual(len(logps), 9)

        for ind in inds:
            self.assertEqual(ind.keys(), {'a'})

        for _logp in logps:
            self.assertIsNotNone(_logp.grad_fn)

    def test_gmm_1d_normals(self):
        t.distributions.Distribution.set_default_validate_args(False)
        gmp = GaussianMixturePopulation({'a': t.Size((30, 50))}, t.Size((7,)), lambda x: x, 0.0)
        inds, logps = zip(*gmp.sample(9))

        self.assertEqual(len(t.unique(inds[0]['a'])), 7)

    def test_gmm_2d_normals(self):
        t.distributions.Distribution.set_default_validate_args(False)
        gmp = GaussianMixturePopulation({'a': t.Size((30, 50))}, t.Size((7, 2)), lambda x: x, 0.0)

        inds, logps = zip(*gmp.sample(9))

        self.assertEqual(len(t.unique(inds[0]['a'].reshape((-1, 2)), dim=0)), 7)
