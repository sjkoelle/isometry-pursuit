import unittest

import numpy as np

from isometry_pursuit.algorithm import forward_backward
from isometry_pursuit.alternating_search import alternating_add_delete
from isometry_pursuit.experiment import _gl_loss


class AlternatingSearchTests(unittest.TestCase):
    def test_preserves_cardinality_rank_and_monotonic_loss(self):
        rng = np.random.default_rng(9)
        matrix = rng.normal(size=(3, 10))
        initial = [0, 1, 2]
        result = alternating_add_delete(
            matrix, _gl_loss, initial, expansion_width=2, max_cycles=4,
            random_state=12,
        )
        selected = result["selected_indices"]
        self.assertEqual(len(selected), 3)
        self.assertEqual(np.linalg.matrix_rank(matrix[:, selected]), 3)
        self.assertLessEqual(result["loss"], _gl_loss(matrix[:, initial]))
        accepted_losses = [step["proposed_loss"] for step in result["history"] if step["accepted"]]
        self.assertEqual(accepted_losses, sorted(accepted_losses, reverse=True))

    def test_expansion_path_differs_from_exhaustive_swap(self):
        matrix = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, -1.0]])
        values = {
            frozenset((0, 1)): 5.0,
            frozenset((0, 1, 2)): 0.0,
            frozenset((0, 1, 3)): 1.0,
            frozenset((0, 1, 2, 3)): 0.0,
            frozenset((0, 2, 3)): -1.0,
            frozenset((1, 2, 3)): 9.0,
            frozenset((0, 2)): 10.0,
            frozenset((1, 2)): 10.0,
            frozenset((0, 3)): 1.0,
            frozenset((1, 3)): 4.0,
            frozenset((2, 3)): 10.0,
        }

        def lookup(_, indices=None):
            return values[frozenset(int(index) for index in indices)]

        width_one = alternating_add_delete(
            matrix, lookup, [0, 1], expansion_width=1, max_cycles=1, random_state=0
        )
        width_two = alternating_add_delete(
            matrix, lookup, [0, 1], expansion_width=2, max_cycles=1, random_state=0
        )
        swapped = forward_backward(
            matrix, lookup, target_dimension=2, selected_indices=[0, 1],
            max_rounds=1, random_state=0,
        )
        self.assertEqual(set(width_one["selected_indices"]), {0, 1})
        self.assertEqual(set(width_two["selected_indices"]), {0, 3})
        self.assertEqual(set(swapped), {0, 3})
        self.assertGreater(width_two["candidate_evaluations"], 0)

    def test_no_external_columns_is_a_noop(self):
        result = alternating_add_delete(np.eye(3), _gl_loss, [0, 1, 2])
        self.assertEqual(result["selected_indices"], [0, 1, 2])
        self.assertEqual(result["cycles"], 0)

    def test_validation(self):
        matrix = np.eye(3)
        with self.assertRaises(ValueError):
            alternating_add_delete(matrix, _gl_loss, [0, 1], target_dimension=3)
        with self.assertRaises(ValueError):
            alternating_add_delete(matrix, _gl_loss, [0, 1, 2], expansion_width=0)
        with self.assertRaises(ValueError):
            alternating_add_delete(matrix, _gl_loss, [0, 1, 2], max_cycles=0)
        with self.assertRaises(ValueError):
            alternating_add_delete(matrix, _gl_loss, [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
