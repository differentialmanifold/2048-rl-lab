import random

import numpy as np
import pytest

from algorithms.mcts_chance import ChanceMCTS
from gym2048_env import Gym2048Env


def initial_board():
    state = np.zeros((4, 4), dtype=np.int32)
    state[0, 0] = 2
    return state


def test_search_counts_legal_actions_and_multiple_outcomes_without_mutating_input():
    state = initial_board()
    original = state.copy()
    planner = ChanceMCTS(100, seed=4)
    action, policy = planner.search(state)
    np.testing.assert_array_equal(state, original)
    assert action in (2, 3)
    assert policy.shape == (4,) and policy.sum() == pytest.approx(1.)
    np.testing.assert_array_equal(policy[:2], [0., 0.])
    root = planner.last_root
    assert root.visits == 100 and sum(edge.visits for edge in root.actions.values()) == 100
    for action, edge in root.actions.items():
        assert len(edge.outcomes) > 1
        assert sum(node.visits for node in edge.outcomes.values()) == edge.visits
        assert policy[action] == pytest.approx(edge.visits / 100)
    assert planner.last_stats['simulations'] == 100
    assert planner.last_stats['rollout_transitions'] > 0


def test_resampling_preserves_actual_tile_value_and_location_probabilities(monkeypatch):
    # Short-circuit only the rollout tail, so thousands of root samples are
    # inexpensive; tree traversal still uses the real Gym spawn implementation.
    monkeypatch.setattr(ChanceMCTS, '_rollout', lambda self, env, info: info['score'])
    planner = ChanceMCTS(4000, seed=51)
    planner.search(initial_board())
    fours = total = 0
    for action, edge in planner.last_root.actions.items():
        fixed_position = (0, 3) if action == 2 else (3, 0)
        locations = np.zeros((4, 4), dtype=int)
        for node in edge.outcomes.values():
            board = node.matrix.copy()
            assert board[fixed_position] == 2
            board[fixed_position] = 0
            row, col = np.argwhere(board != 0)[0]
            value = board[row, col]
            assert np.count_nonzero(board) == 1 and value in (2, 4)
            locations[row, col] += node.visits
            fours += node.visits * (value == 4)
            total += node.visits
        assert locations[fixed_position] == 0
        frequencies = np.delete(locations.flatten(), np.ravel_multi_index(fixed_position, (4, 4)))
        np.testing.assert_allclose(frequencies / edge.visits, np.full(15, 1/15), atol=.025, rtol=0)
    assert total == 4000 and fours / total == pytest.approx(.1, abs=.025)


class LotteryEnv:
    """One action, two terminal outcomes: payoff 10 with p=.1, otherwise 1."""
    def __init__(self, size, matrix, seed):
        self.rng = random.Random(seed)
        self.state = np.array(matrix, copy=True)
        self.info = dict(can_move_dir=[True, False, False, False], score=0.)

    def _get_info(self):
        return self.info

    def reset_to_matrix(self, matrix):
        self.state = np.array(matrix, copy=True)
        self.info = dict(can_move_dir=[True, False, False, False], score=0.)
        return self.state.copy(), self.info

    def step(self, action):
        assert action == 0
        payoff = 10. if self.rng.random() < .1 else 1.
        self.state[0, 0] = int(payoff)
        self.info = dict(can_move_dir=[False] * 4, score=payoff)
        return self.state.copy(), 0., True, False, self.info

    def close(self):
        pass


def test_chance_values_average_sample_frequencies_not_distinct_outcomes_or_max(monkeypatch):
    monkeypatch.setattr('algorithms.mcts_chance.Gym2048Env', LotteryEnv)
    planner = ChanceMCTS(2000, seed=72)
    planner.search(initial_board())
    edge = planner.last_root.actions[0]
    assert len(edge.outcomes) == 2
    weighted_mean = sum(node.matrix[0, 0] * node.visits for node in edge.outcomes.values()) / edge.visits
    assert edge.mean_value == pytest.approx(weighted_mean)
    assert edge.mean_value == pytest.approx(1.9, abs=.25)
    assert planner.last_stats['tree_transitions'] == 2000
    assert planner.last_stats['rollout_transitions'] == 0


def test_search_rng_is_local_reproducible_and_does_not_change_real_spawns():
    py_state, np_state = random.getstate(), np.random.get_state()
    action, policy = ChanceMCTS(30, seed=7).search(initial_board())
    assert random.getstate() == py_state
    current_np = np.random.get_state()
    np.testing.assert_array_equal(current_np[1], np_state[1])
    assert current_np[2:] == np_state[2:]
    random.random()
    np.random.random(100)
    repeated_action, repeated_policy = ChanceMCTS(30, seed=7).search(initial_board())
    assert action == repeated_action
    np.testing.assert_array_equal(policy, repeated_policy)
    real, reference = Gym2048Env(), Gym2048Env()
    state, _ = real.reset(seed=11)
    reference.reset(seed=11)
    planner = ChanceMCTS(8, seed=5)
    for _ in range(5):
        action, _ = planner.search(state)
        state, reward, done, _, _ = real.step(action)
        expected, expected_reward, expected_done, _, _ = reference.step(action)
        np.testing.assert_array_equal(state, expected)
        assert reward == expected_reward and done == expected_done
    real.close()
    reference.close()


def test_terminal_root_and_invalid_budgets():
    terminal = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    planner = ChanceMCTS(8)
    action, policy = planner.search(terminal)
    assert action == 0 and not policy.any()
    assert planner.last_stats['simulations'] == 0
    with pytest.raises(ValueError, match='positive'):
        ChanceMCTS(0)
    with pytest.raises(ValueError, match='nonnegative'):
        ChanceMCTS(exploration=-1)


def test_new_agent_runs_a_complete_evaluation_game():
    from evaluate import evaluate_game
    result, metadata = evaluate_game(('mcts', None, 1, 9000, 'cpu'))
    assert result['steps'] > 0
    assert result['board_sum'] - result['spawn_return'] in (2, 4)
    assert metadata['search_budget'] == 1
    assert metadata['transition_model'] == 'resampled_chance'




