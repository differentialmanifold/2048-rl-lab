import json

import numpy as np
import pytest
import torch
from torch import nn

from algorithms import muzero
from common.checkpoints import export_model, model_from_checkpoint, read_checkpoint, save_checkpoint
from common.models import ARCHITECTURES, preprocess_observation
from common.parallel import GamePool, model_snapshot, worker_model
from common.training import setup


def tiny_history():
    return muzero.GameHistory(
        observations=np.array([[1] + [0] * 15, [1, 1] + [0] * 14, [2, 1] + [0] * 14,
                               [2, 2] + [0] * 14], dtype=np.float32),
        actions=np.array([2, 3, 0]), rewards=np.array([2., 4., 2.], dtype=np.float32),
        policies=np.array([[0., 0., .75, .25], [.1, 0., 0., .9], [.6, .4, 0., 0.]], dtype=np.float32),
        root_values=np.array([10., 20., 30.], dtype=np.float32))


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_network_recurrent_shapes_determinism_and_serialization(tmp_path, architecture):
    setup(3)
    model = muzero.MuZeroNetwork(architecture=architecture, latent_dim=32)
    x = torch.arange(16).float().repeat(2, 1)
    hidden = model.h(x)
    assert hidden.shape == (2, 32)
    next_hidden, reward = model.g(hidden, torch.tensor([0, 2]))
    assert next_hidden.shape == (2, 32) and reward.shape == (2,)
    policy, value = model.f(next_hidden)
    assert policy.shape == (2, 4) and value.shape == (2,)
    torch.testing.assert_close(model.h(x[0]), hidden[0])
    torch.testing.assert_close(model.g(hidden, torch.tensor([0, 2]))[0], next_hidden, atol=0, rtol=0)
    clone = worker_model(model_snapshot(model))
    torch.testing.assert_close(clone.h(x), hidden, atol=0, rtol=0)
    checkpoint = tmp_path / 'model.pt'
    save_checkpoint(checkpoint, model, torch.optim.Adam(model.parameters()), 1, 'muzero', {}, 0.)
    restored = model_from_checkpoint(read_checkpoint(checkpoint))
    torch.testing.assert_close(restored.g(hidden, torch.tensor([0, 2]))[0], next_hidden, atol=0, rtol=0)


class KnownModel(nn.Module):
    """All leaves have value 5; action a produces raw reward 128*(a+1)."""
    def __init__(self):
        super().__init__()
        self.h_calls = self.g_calls = 0

    def h(self, observation):
        self.h_calls += 1
        return torch.tensor([0.])

    def g(self, hidden, action):
        self.g_calls += 1
        return hidden + 1, torch.tensor(128. * (action + 1))

    def f(self, hidden):
        return torch.zeros(4), torch.tensor(5.)


def test_search_uses_only_latents_and_backs_up_rewards_once(monkeypatch):
    from board import Board
    from gym2048_env import Gym2048Env
    def forbidden(*args, **kwargs):
        raise AssertionError('MuZero search must never simulate a real board')
    monkeypatch.setattr(Board, 'step', forbidden)
    monkeypatch.setattr(Gym2048Env, 'step', forbidden)
    model = KnownModel()
    planner = muzero.MCTS(model, 'cpu', gamma=.5, dirichlet_frac=0.)
    root = planner.search(np.zeros((4, 4)), [False, False, True, False], 2)
    assert root.visit_count == 2 and list(root.children) == [2]
    # Simulation 1: 3+.5*5=5.5; simulation 2: 3+.5*(1+.5*5)=4.75.
    assert root.value == pytest.approx(5.125)
    assert root.children[2].value == pytest.approx(4.25)
    assert len(root.children[2].children) == 4  # No simulator-based internal masking.
    assert model.h_calls == 1 and model.g_calls == 2
    assert model.training  # Inference restores the previous mode.
    np.testing.assert_array_equal(muzero.visit_policy(root), [0, 0, 1, 0])
    with pytest.raises(ValueError, match='terminal'):
        planner.search(np.zeros((4, 4)), [False] * 4, 2)


def test_depth_limit_bootstraps_and_cached_edges_do_not_rerun_dynamics():
    model = KnownModel()
    root = muzero.MCTS(model, 'cpu', gamma=.5, search_depth=1, dirichlet_frac=0.).search(
        np.zeros((4, 4)), [False, False, True, False], 20)
    assert root.value == pytest.approx(5.5)
    assert root.children[2].visit_count == 20
    assert model.g_calls == 1


def test_search_noise_is_local_and_actual_decisions_are_legal():
    setup(8)
    model = muzero.MuZeroNetwork()
    state = np.zeros((4, 4), dtype=int)
    state[0, 0] = 2
    mask = [False, False, True, True]
    rng = np.random.get_state()
    first = muzero.MCTS(model, 'cpu', seed=9).search(state, mask, 16)
    np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
    np.random.random(100)
    second = muzero.MCTS(model, 'cpu', seed=9).search(state, mask, 16)
    np.testing.assert_array_equal(muzero.visit_policy(first), muzero.visit_policy(second))
    assert muzero.search_controller(model, simulations=8)(state, {'can_move_dir': mask}) in (2, 3)


def test_targets_nstep_and_terminal_absorption():
    history = tiny_history()
    assert history.target_value(0, 2, .5, 1.) == pytest.approx(2/128 + .5*4/128 + .25*30)
    assert history.target_value(1, 2, .5, 1.) == pytest.approx(4/128 + .5*2/128)
    assert history.target_value(3, 10, 1.) == 0
    batch = muzero.make_batch([(history, 1)], 4, 10, 1., 'cpu', td_lambda=1.)
    torch.testing.assert_close(batch.values, torch.tensor([[6/128, 2/128, 0., 0., 0.]]))
    torch.testing.assert_close(batch.rewards, torch.tensor([[4., 2., 0., 0.]]))
    torch.testing.assert_close(batch.actions[0, :2], torch.tensor([3, 0]))
    torch.testing.assert_close(batch.policy_mask, torch.tensor([[True, True, False, False, False]]))
    assert batch.policies[0, 2:].sum() == 0
    assert ((batch.actions >= 0) & (batch.actions < 4)).all()


@pytest.mark.parametrize('td_steps', [1, 10])
@pytest.mark.parametrize('td_lambda', [0., .5, 1.])
def test_lambda_returns_match_weighted_nstep_returns_and_alphazero(td_steps, td_lambda):
    from algorithms.alphazero import td_lambda_targets
    length, gamma = 15, .999
    history = muzero.GameHistory(np.zeros((length + 1, 16)), np.zeros(length, dtype=int),
                                np.array([2., 4., 2.] * 5), np.full((length, 4), .25),
                                np.linspace(3., 15., length))
    actual = np.array([history.target_value(t, td_steps, gamma, td_lambda) for t in range(length)])
    # Independently expand the weighted 1..n-step returns, including the final tail weight.
    for t in range(length):
        horizon = min(td_steps, length - t)
        expected = 0.
        for k in range(1, horizon + 1):
            nstep = sum(gamma**j * history.rewards[t + j] / 128 for j in range(k))
            if t + k < length:
                nstep += gamma**k * history.root_values[t + k]
            weight = td_lambda**(k - 1) * (1 - td_lambda if k < horizon else 1.)
            expected += weight * nstep
        assert actual[t] == pytest.approx(expected)
    np.testing.assert_allclose(actual, td_lambda_targets(history.rewards / 128,
        np.append(history.root_values, 0.), td_steps, gamma, td_lambda), rtol=1e-6)
    assert actual[-1] == pytest.approx(history.rewards[-1] / 128)


def test_sampled_batch_applies_lambda_to_every_value_target(monkeypatch):
    history = tiny_history()
    monkeypatch.setattr(np.random, 'randint', lambda high, size=None: np.zeros(size, dtype=int)
                        if size is not None else 0)
    batch = muzero.sample_batch([history], 1, 4, 10, .999, 'cpu', .5)
    last = 2/128
    middle = 4/128 + .999 * (.5*30 + .5*last)
    first = 2/128 + .999 * (.5*20 + .5*middle)
    torch.testing.assert_close(batch.values, torch.tensor([[first, middle, last, 0., 0.]]))
    torch.testing.assert_close(batch.rewards, torch.tensor([[2., 4., 2., 0.]]))
    assert history.target_value(0) == pytest.approx(first)


def test_replay_sampling_is_transition_uniform_and_does_not_cross_games(monkeypatch):
    first, second = tiny_history(), tiny_history()
    second.observations += 8
    second.rewards += 10
    monkeypatch.setattr(np.random, 'randint', lambda high, size=None: np.array([0, 2, 3, 5])
                        if size is not None else 1)
    batch = muzero.sample_batch([first, second], 4, 2, 10, 1., 'cpu')
    torch.testing.assert_close(batch.observations[:, 0], torch.tensor([1., 2., 9., 10.]))
    torch.testing.assert_close(batch.rewards, torch.tensor([[2., 4.], [2., 0.], [12., 14.], [12., 0.]]))


def test_recurrent_loss_trains_h_g_f_without_future_observation_inputs():
    setup(4)
    model = muzero.MuZeroNetwork(latent_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batch = muzero.make_batch([(tiny_history(), 0), (tiny_history(), 2)], 4, 10, 1., 'cpu', td_lambda=1.)
    # Only future policies contribute: gradient to h must pass through g.
    batch.policy_mask[:, 0] = False
    calls = []
    hook = model.encoder.embedding.register_forward_hook(lambda *_: calls.append(1))
    losses = muzero.train_batch(model, optimizer, batch)
    hook.remove()
    assert len(calls) == 1
    assert all(np.isfinite(value) for value in losses.values())
    for parameter in (model.encoder.embedding.weight, model.representation[0].weight,
                      model.dynamics[0].weight, model.next_latent[0].weight,
                      model.policy_head.weight, model.value_head.weight, model.reward_head.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_real_selfplay_and_parallel_evaluation_match_serial():
    setup(2)
    model = muzero.MuZeroNetwork(latent_dim=16)
    with GamePool(1) as serial, GamePool(2) as parallel:
        args = (model, 'cpu', 2, 2, 10, 27, 1., 3)
        whole = muzero.collect_self_play(*args, serial)
        split = muzero.collect_self_play(*args, parallel)
        for (a, a_info), (b, b_info) in zip(whole, split):
            assert a_info == b_info
            for key in vars(a):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            assert len(a.observations) == len(a) + 1
            assert set(a.rewards).issubset({2., 4.})
            raw_board = np.where(a.observations[-1] > 0, 2 ** a.observations[-1], 0)
            from board import Board
            assert not any(Board(raw_board).can_move_dir)
        first = muzero.evaluate_search(model, 2, 71, 2, search_depth=3, pool=serial)
        second = muzero.evaluate_search(model, 2, 71, 2, search_depth=3, pool=parallel)
        assert first['results'] == second['results']


@pytest.mark.parametrize('workers', [1, 2])
def test_full_training_resume_replay_optimizer_and_export(tmp_path, monkeypatch, workers):
    import plot
    from play import make_agent
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--device', 'cpu', '--self-play-games', '2', '--mcts-sims', '2', '--train-steps', '2', '--batch-size', '8',
               '--latent-dim', '16', '--unroll-steps', '3', '--search-depth', '3',
               '--eval-every', '1', '--eval-episodes', '2', '--eval-mcts-sims', '2', '--workers', str(workers)]
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    muzero.main(['--iterations', '2', '--save-dir', str(whole), *options])
    muzero.main(['--iterations', '1', '--save-dir', str(split), *options])
    muzero.main(['--iterations', '2', '--resume', str(split / 'last.pt'), '--workers', str(workers)])
    expected, resumed = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert expected['config']['td_steps'] == resumed['config']['td_steps'] == 10
    assert expected['config']['td_lambda'] == resumed['config']['td_lambda'] == .5
    assert expected['config']['gamma'] == resumed['config']['gamma'] == .999
    assert resumed['reward_objective'] == 'spawn_mass'
    assert resumed['iteration'] == 2 and len(resumed['replay']) == 4
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
    for key, value in expected['optimizer']['state'].items():
        for name in value:
            torch.testing.assert_close(value[name], resumed['optimizer']['state'][key][name], atol=0, rtol=0)
    for left, right in zip(expected['replay'], resumed['replay']):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    rows = [json.loads(line) for line in (split / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['updates'] == 2 and row['architecture'] == 'cnn2x2' for row in rows)
    assert all((row['td_steps'], row['td_lambda'], row['gamma']) == (10, .5, .999) for row in rows)
    exported = tmp_path / 'export.pt'
    export_model(split / 'last.pt', exported)
    data = torch.load(exported, weights_only=True)
    assert 'replay' not in data and data['config']['search_depth'] == 3
    assert data['config']['gamma'] == .999
    controller, label, metadata = make_agent('muzero', exported, budget=2, seed=42)
    state = np.zeros((4, 4), dtype=int)
    state[0, 0] = 2
    assert controller(state, {'can_move_dir': [False, False, True, True]}) in (2, 3)
    assert 'MUZERO' in label and metadata['search_budget'] == 2
    with pytest.raises(ValueError, match='full training checkpoint'):
        muzero.main(['--iterations', '3', '--resume', str(exported)])
    with pytest.raises(SystemExit):
        muzero.main(['--iterations', '3', '--resume', str(split / 'last.pt'), '--latent-dim', '32'])
    with pytest.raises(SystemExit):
        muzero.main(['--iterations', '3', '--resume', str(split / 'last.pt'), '--search-depth', '4'])
    for flag, value in [('--td-steps', '0'), ('--td-lambda', '1')]:
        with pytest.raises(SystemExit):
            muzero.main(['--iterations', '3', '--resume', str(split / 'last.pt'), flag, value])




@pytest.mark.parametrize('flag', ['--unroll-steps', '--search-depth', '--buffer-games', '--latent-dim'])
def test_positive_arguments(flag):
    with pytest.raises(SystemExit):
        muzero.main(['--iterations', '1', flag, '0'])


@pytest.mark.parametrize('value', ['-0.1', '1.1', 'nan'])
def test_lambda_argument_range(value):
    with pytest.raises(SystemExit):
        muzero.main(['--iterations', '1', '--td-lambda', value])
