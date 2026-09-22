"""Spawn reward, finite-horizon lambda mixtures, and checkpoint compatibility."""
import numpy as np
import pytest
import torch

from algorithms import alphazero
from common.models import ActorCritic
from common.checkpoints import save_checkpoint, read_checkpoint, export_model
from common.evaluation import summarize_results
from common.training import setup


def test_td_lambda_is_mixture_of_n_step_returns_and_stops_at_ten():
    rewards = np.arange(1, 15, dtype=float)
    values = np.linspace(2., 9., len(rewards) + 1)
    values[-1] = 0.
    gamma, lam = .999, .5
    actual = alphazero.td_lambda_targets(rewards, values)
    expected = []
    for t in range(len(rewards)):
        horizon = min(10, len(rewards) - t)
        total = 0.
        for n in range(1, horizon + 1):
            g = sum(gamma**j * rewards[t+j] for j in range(n)) + gamma**n * values[t+n]
            weight = lam**(n-1) * ((1-lam) if n < horizon else 1.)
            total += weight*g
        expected.append(total)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    changed = rewards.copy();changed[10:] += 10000
    assert alphazero.td_lambda_targets(changed, values)[0] == actual[0]


def test_td_lambda_endpoints_terminal_and_truncation():
    rewards, values = [1., 2., 3.], [99., 4., 5., 0.]
    np.testing.assert_allclose(alphazero.td_lambda_targets(rewards, values, gamma=.9, td_lambda=0.),
                               [4.6, 6.5, 3.])
    np.testing.assert_allclose(alphazero.td_lambda_targets(rewards, values, gamma=.9, td_lambda=1.),
                               [5.23, 4.7, 3.])
    # Two-step horizon must leave the last-state bootstrap intact, not become MC.
    assert alphazero.td_lambda_targets(rewards, values, 2, .9, 1.)[0] == pytest.approx(6.85)
    assert alphazero.td_lambda_targets([2.], [99., 8.], gamma=.9)[0] == pytest.approx(9.2)
    assert alphazero.td_lambda_targets([2.], [99., 0.], gamma=.9)[0] == 2.


@pytest.mark.parametrize('options', [{'td_steps': 0}, {'td_lambda': -1}, {'td_lambda': 1.1}])
def test_td_lambda_rejects_invalid_parameters(options):
    with pytest.raises(ValueError):
        alphazero.td_lambda_targets([1.], [2., 0.], **options)


def test_self_play_uses_spawn_reward_and_future_search_values(monkeypatch):
    rewards = [2., 4., 2.]
    setup(0)
    class Game:
        def reset(self, seed):
            self.t = 0
            return np.full((4, 4), 2), {'can_move_dir': [True]*4}
        def step(self, action):
            self.t += 1
            return (np.full((4, 4), 2**(self.t+1)), [2., 4., 2.][self.t-1], self.t == 3, False,
                    {'can_move_dir': [True]*4, 'merge_reward': [4., 16., 0.][self.t-1],
                     'merge_score': [4., 20., 20.][self.t-1], 'max_value': 8})
        def close(self):
            pass
    search_values = iter([10., 20., 30.])
    def search(self, root, sims):
        root.children = {0: alphazero.Node(1., visit_count=sims)}
        root.visit_count = sims
        root.value_sum = next(search_values)*sims
    monkeypatch.setattr(alphazero, 'Gym2048Env', Game)
    monkeypatch.setattr(alphazero.MCTS, 'run', search)
    examples, info = alphazero.self_play_game(ActorCritic(), 'cpu', 2, 30)
    g2 = rewards[2]/128
    g1 = rewards[1]/128 + .999 * (.5*30 + .5*g2)
    g0 = rewards[0]/128 + .999 * (.5*20 + .5*g1)
    np.testing.assert_allclose([ex.value for ex in examples], [g0, g1, g2], rtol=1e-6)
    assert info['spawn_return'] == 8. and info['merge_score'] == 20.






def test_export_and_resume_preserve_objective_and_targets(tmp_path):
    setup(0)
    model = ActorCritic()
    source, exported = tmp_path/'source.pt', tmp_path/'export.pt'
    save_checkpoint(source, model, torch.optim.Adam(model.parameters()), 1, 'alphazero',
                    dict(gamma=.999, td_steps=10, td_lambda=.5), 24.)
    export_model(source, exported)
    assert read_checkpoint(exported)['reward_objective'] == 'spawn_mass'
    from play import make_agent
    _, _, metadata = make_agent('alphazero', exported, budget=1)
    assert metadata['reward_objective'] == 'spawn_mass'
    with pytest.raises(SystemExit):
        alphazero.main(['--resume', str(source), '--iterations', '2', '--td-lambda', '1.',
                       '--save-dir', str(tmp_path/'changed')])
    assert not (tmp_path/'changed').exists()


def test_play_search_receives_checkpoint_reward_and_gamma(tmp_path, monkeypatch):
    from play import make_agent
    setup(0)
    model = ActorCritic()
    path = tmp_path/'spawn.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 'alphazero',
                    dict(gamma=.999, td_steps=10, td_lambda=.5), 0.)
    captured = []
    def run(self, root, sims):
        captured.append((self.gamma, sims))
        root.children = {0: alphazero.Node(1., visit_count=1)}
    monkeypatch.setattr(alphazero.MCTS, 'run', run)
    action, _, metadata = make_agent('alphazero', path, budget=100, seed=42)
    assert action(np.full((4,4),2), {'can_move_dir':[True]*4}) == 0
    assert captured == [(.999, 100)]
    assert metadata['reward_objective'] == 'spawn_mass'


def test_search_validation_uses_spawn_selection_metric():
    setup(0)
    result = alphazero.evaluate_search(ActorCritic(), 1, 11, 2)
    assert result['score_metric'] == 'spawn_mass'
    assert result['mean_return'] == result['mean_spawn_return']
    assert result['mean_spawn_return'] == result['results'][0]['spawn_return']
