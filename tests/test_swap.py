import itertools
import unittest

import numpy as np

from isometry_pursuit.algorithm import greedy_swap
from isometry_pursuit.experiment import _gl_loss, compare_swap_initializations


class SwapTests(unittest.TestCase):
    def test_four_methods_preserve_rank_and_restrict_ip(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(2, 7))
        ip = [6, 2, 4, 0]
        results = compare_swap_initializations(X, ip, random_state=4)
        self.assertEqual(set(results), {'greedy_swap', 'backward_swap',
                                      'ip_greedy_swap', 'ip_backward_swap'})
        for name, result in results.items():
            selected = result['indices']
            self.assertEqual(len(set(selected)), 2)
            self.assertEqual(np.linalg.matrix_rank(X[:, selected]), 2)
            self.assertLessEqual(result['loss'], result['initial_loss'] + 1e-12)
            if name.startswith('ip_'):
                self.assertTrue(set(selected).issubset(ip))
            candidates = ip if name.startswith('ip_') else range(7)
            for removed, added in itertools.product(selected, candidates):
                if added in selected:
                    continue
                trial = [j for j in selected if j != removed] + [added]
                self.assertGreaterEqual(_gl_loss(X[:, trial]), result['loss'] - 1e-12)

    def test_initialization_wrapper(self):
        X = np.array([[1., 0., 1.], [0., 1., .2]])
        for initialization in ('forward', 'backward'):
            selected = greedy_swap(X, _gl_loss, initialization=initialization, random_state=0)
            self.assertEqual(len(selected), 2)
            self.assertTrue(np.isfinite(_gl_loss(X[:, selected])))
        with self.assertRaises(ValueError):
            greedy_swap(X, _gl_loss, initialization='bad')

    def test_insufficient_ip_support(self):
        results = compare_swap_initializations(np.eye(3), [0, 1])
        self.assertEqual(results['ip_backward_swap']['status'], 'insufficient_ip_support')
        self.assertIsNone(results['ip_greedy_swap']['loss'])
        self.assertEqual(results['greedy_swap']['status'], 'ok')

    def test_reproducibility_and_input_validation(self):
        X = np.array([[1., 0., 1.], [0., 1., 1.]])
        first = compare_swap_initializations(X, [0, 1, 2], random_state=20)
        second = compare_swap_initializations(X, [0, 1, 2], random_state=20)
        for name in first:
            self.assertEqual(first[name]['indices'], second[name]['indices'])
        with self.assertRaises(ValueError):
            compare_swap_initializations(X, [0, 0, 2])


if __name__ == '__main__':
    unittest.main()
