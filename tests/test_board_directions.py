"""Compare direct row/column moves against an independent rotation-based oracle."""
from itertools import product
import random
import numpy as np
import pytest
from board import Board


def reference_afterstate(matrix, direction):
    rotated = np.rot90(matrix, direction).copy()
    score = 0
    for row in rotated:
        tiles = [int(value) for value in row if value]
        output = []
        while tiles:
            value = tiles.pop(0)
            if tiles and tiles[0] == value:
                value += tiles.pop(0)
                score += value
            output.append(value)
        row[:] = output + [0] * (len(row) - len(output))
    result = np.rot90(rotated, -direction).copy()
    return result, score, not np.array_equal(result, matrix)


class ReferenceBoard(Board):
    def check_swipe_all_direction(self):
        return [reference_afterstate(self.matrix, direction)[2] for direction in range(4)]

    def move(self, direction):
        self.reward = 0
        self.matrix, self.merge_reward, self.has_changed = reference_afterstate(self.matrix, direction)
        if self.has_changed:
            self.add_random_tile()
        self.last_action = direction
        self.can_move_dir = self.check_swipe_all_direction()
        self.merge_score += self.merge_reward


def assert_afterstates(matrix):
    expected_masks = []
    for direction in range(4):
        expected, score, changed = reference_afterstate(matrix, direction)
        board = Board(matrix, size=len(matrix))
        before_rng = random.getstate()
        board.add_random_tile = lambda: None
        board.move(direction)
        np.testing.assert_array_equal(board.matrix, expected)
        assert board.merge_reward == score
        assert board.merge_score == score
        assert board.has_changed == changed
        assert random.getstate() == before_rng
        expected_masks.append(changed)
    board = Board(matrix, size=len(matrix))
    original_array = board.matrix
    assert board.check_swipe_all_direction() == expected_masks
    assert board.can_swipe_left() == expected_masks[0]
    assert board.matrix is original_array
    np.testing.assert_array_equal(board.matrix, matrix)


def test_exhaustive_short_lines_all_directions():
    # Includes gaps, triple/quadruple equal tiles, and adjacent possible merges.
    for line in product((0, 2, 4, 8, 16), repeat=4):
        matrix = np.tile(line, (4, 1))
        assert_afterstates(matrix)


@pytest.mark.parametrize('size', [1, 2, 3, 4, 5])
def test_random_boards_different_sizes_and_large_tiles(size):
    rng = np.random.default_rng(413)
    for _ in range(80):
        matrix = rng.choice([0, 0, 0, 2, 4, 8, 16, 2048, 32768, 65536], (size, size))
        assert_afterstates(matrix)


def test_fixed_seed_complete_episodes_match_rotation_reference():
    for seed in range(20):
        actual = Board(rng=random.Random(seed))
        expected = ReferenceBoard(rng=random.Random(seed))
        actions = random.Random(seed + 1000)
        for step in range(5000):
            np.testing.assert_array_equal(actual.matrix, expected.matrix)
            assert actual.can_move_dir == expected.can_move_dir
            if actual.has_done():
                break
            # Include illegal actions to verify they do not consume spawn RNG.
            action = actions.randrange(4)
            result = actual.step(action)
            reference = expected.step(action)
            np.testing.assert_array_equal(result[0], reference[0])
            assert result[1:] == reference[1:]
            assert actual.merge_reward == expected.merge_reward
            assert actual.merge_score == expected.merge_score
            assert actual.has_changed == expected.has_changed
            assert actual.last_action == expected.last_action
            assert actual.rng.getstate() == expected.rng.getstate()
        else:
            pytest.fail('Reference episode did not terminate')


def test_moves_masks_and_search_do_not_call_rot90(monkeypatch):
    from algorithms.alphazero import MCTS, Node
    from common.models import ActorCritic
    from common.training import setup
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('Game operations must not rotate the board')
    monkeypatch.setattr(np, 'rot90', forbidden)
    matrix = np.array([[2, 2, 4, 0], [0, 4, 8, 8], [16, 0, 2, 2], [0, 0, 0, 0]])
    for direction in range(4):
        board = Board(matrix, rng=random.Random(3))
        board.step(direction)
        board.check_swipe_all_direction()
    setup(8)
    root = Node(1., matrix=matrix)
    MCTS(ActorCritic(), torch.device('cpu'), dirichlet_frac=0).run(root, 20)
    assert sum(edge.visit_count for edge in root.children.values()) == 20
