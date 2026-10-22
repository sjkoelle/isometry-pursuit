import unittest

import numpy as np

from isometry_pursuit.experiment import _gl_loss
from isometry_pursuit.lookahead_backward import lookahead_backward_deletion


class LookaheadBackwardTests(unittest.TestCase):
    def test_rank_cardinality_and_explicit_work_counts(self):
        matrix = np.array(
            [[1.0, 0.0, 1.0, 0.2], [0.0, 1.0, 0.1, 1.0]]
        )
        result = lookahead_backward_deletion(
            matrix, _gl_loss, target_dimension=2, depth=2, beam_width=2
        )
        self.assertEqual(len(result["indices"]), 2)
        self.assertEqual(np.linalg.matrix_rank(matrix[:, result["indices"]]), 2)
        # k=4: 4 width-3 plus 2*3 width-2 candidates; k=3: 3 width-2.
        self.assertEqual(result["candidate_evaluations"], 13)
        self.assertEqual(result["arithmetic_work_proxy"], 120)

    def test_depth_two_can_avoid_a_greedy_trap(self):
        matrix = np.ones((1, 4))
        values = {
            (1, 2, 3): 0.0,
            (0, 2, 3): 1.0,
            (0, 1, 3): 2.0,
            (0, 1, 2): 3.0,
            (1, 2): 10.0,
            (1, 3): 11.0,
            (2, 3): 12.0,
            (0, 2): 0.1,
            (0, 3): 0.2,
            (0, 1): 0.3,
        }

        def mapped_loss(sub_matrix, indices=None):
            return values[tuple(int(index) for index in indices)]

        greedy = lookahead_backward_deletion(
            matrix, mapped_loss, target_dimension=2, depth=1, beam_width=2
        )
        lookahead = lookahead_backward_deletion(
            matrix, mapped_loss, target_dimension=2, depth=2, beam_width=2
        )
        self.assertEqual(greedy["loss"], 10.0)
        self.assertEqual(lookahead["loss"], 0.1)

    def test_depth_one_matches_greedy_loss(self):
        rng = np.random.default_rng(18)
        matrix = rng.normal(size=(3, 7))
        result = lookahead_backward_deletion(
            matrix, _gl_loss, target_dimension=3, depth=1, beam_width=5
        )
        support = list(range(matrix.shape[1]))
        while len(support) > 3:
            trials = [support[:i] + support[i + 1 :] for i in range(len(support))]
            support = min(trials, key=lambda trial: (_gl_loss(matrix[:, trial]), trial))
        self.assertAlmostEqual(result["loss"], _gl_loss(matrix[:, support]))

    def test_invalid_parameters_and_infeasible_removals(self):
        with self.assertRaises(ValueError):
            lookahead_backward_deletion(np.eye(2), _gl_loss, depth=0)
        with self.assertRaises(ValueError):
            lookahead_backward_deletion(np.eye(2), _gl_loss, beam_width=0)

        def infeasible(sub_matrix, indices=None):
            return np.inf

        with self.assertRaises(ValueError):
            lookahead_backward_deletion(
                np.ones((1, 2)), infeasible, target_dimension=1
            )


if __name__ == "__main__":
    unittest.main()
