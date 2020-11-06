import unittest
import torch as t

from evostrat.gmm_population import GaussianMixturePopulation


class TestGMMPopulation(unittest.TestCase):

    def test_gmm(self):
        gmp = GaussianMixturePopulation({'a': t.Size((30, 50))}, t.Size((7,)), lambda x: x, 0.0)

        inds, logps = zip(*gmp.sample(9))

        self.assertEqual(len(inds), 9)
        self.assertEqual(len(logps), 9)

        for ind in inds:
            self.assertEqual(ind.keys(), {'a'})

        for _logp in logps:
            self.assertIsNotNone(_logp.grad_fn)

    def test_gmm_1d_normals(self):
        gmp = GaussianMixturePopulation({'a': t.Size((30, 50))}, t.Size((7,)), lambda x: x, 0.0)
        inds, logps = zip(*gmp.sample(9))

        self.assertEqual(len(t.unique(inds[0]['a'])), 7)

    def test_gmm_2d_normals(self):
        gmp = GaussianMixturePopulation({'a': t.Size((30, 50))}, t.Size((7, 2)), lambda x: x, 0.0)

        inds, logps = zip(*gmp.sample(9))

        self.assertEqual(len(t.unique(inds[0]['a'].reshape((-1, 2)), dim=0)), 7)
