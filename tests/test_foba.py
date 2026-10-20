import unittest

import numpy as np

from isometry_pursuit.foba import foba, foba_least_squares


class FoBaTests(unittest.TestCase):
    def test_target_dimension_overrides_thresholds(self):
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(np.eye(4), [3., 2., 1., 0.],
                                       variant=variant, target_dimension=2,
                                       delta=100., epsilon=100., max_iterations=2)
            self.assertEqual(result['support'], [0, 1])
            self.assertEqual(result['status'], 'target_dimension')
            self.assertTrue(result['converged'])

    def test_target_counts_selected_coordinates(self):
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(np.eye(3), np.zeros(3),
                                       variant=variant, target_dimension=3)
            self.assertEqual(len(result['support']), 3)
            self.assertEqual(result['status'], 'target_dimension')
            np.testing.assert_array_equal(result['beta'], np.zeros(3))

    def test_target_after_backward_pruning(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 20))
        X /= np.linalg.norm(X, axis=0)
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(X, X[:, :3].sum(axis=1), variant=variant,
                                       target_dimension=4)
            self.assertTrue(any(e['action'] == 'remove' for e in result['history']))
            self.assertEqual(len(result['support']), 4)
            self.assertEqual(result['status'], 'target_dimension')

    def test_target_cap_and_invalid_sizes(self):
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(np.eye(3), [3., 2., 1.], variant=variant,
                                       target_dimension=3, max_iterations=1)
            self.assertFalse(result['converged'])
            self.assertEqual(result['status'], 'max_iterations')
        for target in (0, -1, 4, 1.5, np.nan, np.inf):
            with self.assertRaises(ValueError):
                foba_least_squares(np.eye(3), np.zeros(3), target_dimension=target)

    def test_exact_sparse_recovery(self):
        for variant in ("obj", "gdt"):
            y = np.array([3., 0., -2., 0.])
            result = foba_least_squares(np.eye(4), y, variant=variant)
            np.testing.assert_allclose(result['beta'], y)
            self.assertEqual(set(result['support']), {0, 2})
            self.assertEqual(result['status'], 'threshold')

    def test_selection_variants_differ_on_unscaled_features(self):
        X, y = np.diag([1., 10.]), np.array([2., 1.])
        obj = foba_least_squares(X, y, max_iterations=1)
        gdt = foba_least_squares(X, y, variant='gdt', max_iterations=1)
        self.assertEqual(obj['support'], [0])
        self.assertEqual(gdt['support'], [1])
        self.assertFalse(obj['converged'])
        self.assertEqual(obj['status'], 'max_iterations')

    def test_thresholds_before_addition_and_strict_boundary(self):
        self.assertEqual(foba_least_squares(np.eye(1), [1.], delta=.6)['support'], [])
        self.assertEqual(foba_least_squares(np.eye(1), [1.], delta=.5)['support'], [0])
        self.assertEqual(foba_least_squares(np.eye(1), [1.], variant='gdt', epsilon=2.)['support'], [])

    def test_backward_deletion_and_support_refits(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 20))
        X /= np.linalg.norm(X, axis=0)
        y = X[:, :3].sum(axis=1)
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(X, y, variant=variant, delta=1e-10)
            removals = [e for e in result['history'] if e['action'] == 'remove']
            self.assertTrue(removals)
            self.assertEqual(set(result['support']), {0, 1, 2})
            for event in result['history']:
                support = event['support']
                beta = np.zeros(20)
                beta[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
                self.assertAlmostEqual(.5 * np.linalg.norm(X @ beta-y)**2, event['objective'])
                if event['action'] == 'remove':
                    self.assertLess(event['deletion_cost'], event['backward_threshold'])

    def test_zero_and_duplicate_columns(self):
        X = np.array([[0., 1., 1.], [0., 0., 0.]])
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(X, [2., 0.], variant=variant)
            self.assertEqual(result['support'], [1])
            self.assertAlmostEqual(result['objective_value'], 0.)

    def test_no_signal(self):
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(np.eye(3), np.zeros(3), variant=variant)
            self.assertEqual(result['support'], [])
            self.assertTrue(result['converged'])

    def test_generic_nonquadratic_objective(self):
        # Separable convex quartic: exercises the generic callback API.
        target = np.array([1., 0., -2.])
        def restricted(support):
            beta = np.zeros(3)
            beta[list(support)] = target[list(support)]
            return beta
        for variant in ('obj', 'gdt'):
            result = foba(lambda b: np.sum((b-target)**4), restricted, 3,
                          variant=variant, gradient=lambda b: 4*(b-target)**3,
                          coordinate_minimizer=lambda b, j: target[j]-b[j])
            np.testing.assert_allclose(result['beta'], target)

    def test_invalid_inputs(self):
        for kwargs in ({'variant': 'bad'}, {'delta': 0}, {'epsilon': np.nan},
                       {'max_iterations': 0}):
            with self.assertRaises(ValueError):
                foba_least_squares(np.eye(2), [1., 0.], **kwargs)
        with self.assertRaises(ValueError):
            foba_least_squares(np.eye(2), [1.])


if __name__ == '__main__':
    unittest.main()
