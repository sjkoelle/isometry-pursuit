import unittest

import numpy as np

from isometry_pursuit.experiment import _gl_loss
from isometry_pursuit.multistart_backward import (
    backward_work_counts,
    multistart_backward_deletion,
    randomized_backward_deletion,
)


class MultiStartBackwardTests(unittest.TestCase):
    def test_cardinality_rank_counts_and_reproducibility(self):
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(3, 8))
        first = multistart_backward_deletion(
            matrix, _gl_loss, num_starts=5, relative_tolerance=0.02, random_state=12
        )
        second = multistart_backward_deletion(
            matrix, _gl_loss, num_starts=5, relative_tolerance=0.02, random_state=12
        )
        evaluations, proxy = backward_work_counts(3, 8, 3)
        self.assertEqual(first["indices"], second["indices"])
        self.assertEqual(len(first["indices"]), 3)
        self.assertEqual(np.linalg.matrix_rank(matrix[:, first["indices"]]), 3)
        self.assertEqual(first["candidate_evaluations"], 5 * evaluations)
        self.assertEqual(first["arithmetic_work_proxy"], 5 * proxy)

    def test_extra_starts_do_not_worsen_baseline_start(self):
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(2, 7))
        result = multistart_backward_deletion(
            matrix, _gl_loss, num_starts=6, relative_tolerance=0.05, random_state=3
        )
        self.assertLessEqual(result["loss"], result["starts"][0]["loss"])

    def test_only_finite_candidates_are_selected(self):
        matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        result = randomized_backward_deletion(
            matrix, _gl_loss, target_dimension=2, relative_tolerance=1.0, random_state=0
        )
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertEqual(np.linalg.matrix_rank(matrix[:, result["indices"]]), 2)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            multistart_backward_deletion(np.eye(2), _gl_loss, num_starts=0)
        with self.assertRaises(ValueError):
            randomized_backward_deletion(np.eye(2), _gl_loss, relative_tolerance=-0.1)
        with self.assertRaises(ValueError):
            randomized_backward_deletion(
                np.eye(2), _gl_loss, selected_indices=[0, 0], target_dimension=1
            )


if __name__ == "__main__":
    unittest.main()
