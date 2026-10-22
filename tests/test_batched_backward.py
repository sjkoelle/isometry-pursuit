import unittest

import numpy as np

from isometry_pursuit.algorithm import backward_elimination
from isometry_pursuit.batched_backward import batched_backward_elimination
from isometry_pursuit.experiment import _gl_loss


class BatchedBackwardTests(unittest.TestCase):
    def test_batch_one_matches_plain_backward_without_ties(self):
        matrix = np.array([[1.0, 0.0, 0.9, 0.2], [0.0, 1.0, 0.1, 0.8]])
        expected = backward_elimination(matrix, _gl_loss, target_dimension=2, random_state=7)
        result = batched_backward_elimination(
            matrix, _gl_loss, target_dimension=2, batch_size=1, random_state=7
        )
        self.assertEqual(set(result["selected_indices"]), set(expected))
        self.assertEqual(result["loss_evaluations"], 4 + 3)

    def test_batch_is_truncated_and_preserves_rank(self):
        matrix = np.array(
            [[1.0, 0.0, 1.0, 1.0, 0.2], [0.0, 1.0, 1.0, 0.1, 1.0]]
        )
        result = batched_backward_elimination(
            matrix, _gl_loss, target_dimension=2, batch_size=4, random_state=2
        )
        selected = result["selected_indices"]
        self.assertEqual(len(selected), 2)
        self.assertEqual(np.linalg.matrix_rank(matrix[:, selected]), 2)
        self.assertEqual(result["history"][-1]["size_after"], 2)
        self.assertGreater(result["rank_checks"], 0)

    def test_halving_schedule_and_initial_subset(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(3, 12))
        result = batched_backward_elimination(
            matrix,
            _gl_loss,
            target_dimension=3,
            selected_indices=list(range(10)),
            batch_size="halving",
            random_state=4,
        )
        self.assertEqual([step["size_after"] for step in result["history"]], [6, 4, 3])
        self.assertTrue(set(result["selected_indices"]).issubset(range(10)))
        self.assertTrue(np.isfinite(_gl_loss(matrix[:, result["selected_indices"]])))

    def test_validation(self):
        matrix = np.eye(2)
        for invalid in (0, -1, 1.5, "other"):
            with self.assertRaises(ValueError):
                batched_backward_elimination(matrix, _gl_loss, batch_size=invalid)
        with self.assertRaises(ValueError):
            batched_backward_elimination(matrix, _gl_loss, target_dimension=3)
        with self.assertRaises(ValueError):
            batched_backward_elimination(
                matrix, _gl_loss, selected_indices=[0, 0], target_dimension=1
            )


if __name__ == "__main__":
    unittest.main()
