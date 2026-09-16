"""Batched trajectory boundaries, spawn workers, and checkpoint continuity."""
import numpy as np
import pytest
import torch
from common.models import ActorCritic
from common.parallel import GamePool
from common.rollout import collect_actor_critic
from common.evaluation import evaluate
from common.training import setup
from common.checkpoints import read_checkpoint, save_checkpoint
from gym2048_env import Gym2048Env
from algorithms import a2c, ppo, alphazero


class ShortGames:
    def reset(self, seed=None):
        self.t = 0
        self.length = seed % 3 + 1
        self.truncate = seed % 2 == 0
        return np.zeros((4, 4)), {'can_move_dir': [True, False, False, False]}

    def step(self, action):
        assert action == 0
        self.t += 1
        end = self.t == self.length
        return np.zeros((4, 4)), 2., end and not self.truncate, end and self.truncate, {
            'can_move_dir': [True, False, False, False], 'max_value': 2}


def test_one_forward_per_state_and_bootstrap_only_for_truncation():
    setup(0)
    model = ActorCritic()
    for p in model.parameters():
        p.data.zero_()
    model.value_head.bias.data.fill_(3.)
    inputs = []
    hook = model.register_forward_pre_hook(lambda _, args: inputs.append(args[0].shape))
    rollout = collect_actor_critic(ShortGames(), model, 3, 1., seed=0, gae_lambda=1.)
    hook.remove()
    # Three games of lengths 1, 2, 3; games 0 and 2 truncate and need one bootstrap.
    assert inputs == [torch.Size([3, 16]), torch.Size([16]), torch.Size([2, 16]),
                      torch.Size([1, 16]), torch.Size([16])]
    assert [e['steps'] for e in rollout.episodes] == [1, 2, 3]
    torch.testing.assert_close(rollout.returns, torch.tensor([
        3 + 2/128, 4/128, 2/128, 3 + 6/128, 3 + 4/128, 3 + 2/128]))


def test_spawn_workers_match_batched_rollouts_and_greedy_evaluation():
    setup(3)
    model = ActorCritic()
    serial = collect_actor_critic(Gym2048Env(), model, 4, .99, 654)
    expected_eval = evaluate(model, 4, 655)
    with GamePool(2) as pool:
        parallel = collect_actor_critic(Gym2048Env(), model, 4, .99, 654, pool=pool)
        parallel_eval = evaluate(model, 4, 655, pool=pool)
    assert serial.episodes == parallel.episodes
    for field in ('states', 'masks', 'actions', 'old_log_probs', 'old_logits', 'advantages', 'returns'):
        torch.testing.assert_close(getattr(serial, field), getattr(parallel, field), atol=2e-6, rtol=2e-6)
    assert expected_eval['results'] == parallel_eval['results']
    assert model.training


def test_parallel_search_games_and_validation_match_serial():
    setup(5)
    model = ActorCritic()
    with GamePool(1) as serial_pool, GamePool(2) as parallel_pool:
        expected = alphazero.collect_self_play(model, torch.device('cpu'), 2, 2, 10, 93, 1., serial_pool)
        actual = alphazero.collect_self_play(model, torch.device('cpu'), 2, 2, 10, 93, 1., parallel_pool)
        for (xs, info), (ys, other_info) in zip(expected, actual):
            assert info == other_info
            assert len(xs) == len(ys)
            for x, y in zip(xs, ys):
                np.testing.assert_array_equal(x.state, y.state)
                np.testing.assert_array_equal(x.policy, y.policy)
                assert x.value == y.value
        first = alphazero.evaluate_search(model, 2, 99, 2, pool=serial_pool)
        second = alphazero.evaluate_search(model, 2, 99, 2, pool=parallel_pool)
        assert first['results'] == second['results']


@pytest.mark.parametrize('trainer', [a2c, ppo, alphazero])
def test_parallel_checkpoint_resume(tmp_path, monkeypatch, trainer):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--workers', '2', '--eval-episodes', '2', '--eval-every', '1', '--seed', '7']
    if trainer is alphazero:
        options += ['--self-play-games', '2', '--mcts-sims', '2', '--train-steps', '1',
                    '--batch-size', '16', '--eval-mcts-sims', '2']
    else:
        options += ['--episodes-per-update', '4']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    trainer.main(['--iterations', '2', '--save-dir', str(whole), *options])
    trainer.main(['--iterations', '1', '--save-dir', str(split), *options])
    trainer.main(['--iterations', '2', '--resume', str(split / 'last.pt'), '--workers', '2'])
    expected, actual = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert actual['config']['workers'] == 2
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)


def test_changed_validation_requires_branch_and_recomputes_best(tmp_path, monkeypatch):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    model = ActorCritic()
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 'alphazero',
                    dict(eval_episodes=20, eval_mcts_sims=200, mcts_sims=200), 99999.)
    options = ['--iterations', '2', '--resume', str(path), '--eval-episodes', '10',
               '--eval-mcts-sims', '100', '--mcts-sims', '100', '--self-play-games', '1',
               '--train-steps', '1']
    with pytest.raises(SystemExit):
        alphazero.main(options)
    calls = []
    def evaluate_stub(model, episodes, seed, sims, gamma, **kwargs):
        calls.append((episodes, sims))
        return {'mean_return': 10.}
    monkeypatch.setattr(alphazero, 'evaluate_search', evaluate_stub)
    monkeypatch.setattr(alphazero, 'collect_self_play', lambda *args: [
        ([alphazero.TrainExample(np.array([[2, 0, 0, 0], *[[0]*4]*3]),
                                  np.array([0., 0., .5, .5]), 1.)],
         {'spawn_return': 128., 'steps': 1, 'max_value': 2})])
    branch = tmp_path / 'branch'
    alphazero.main([*options, '--save-dir', str(branch)])
    assert calls == [(10, 100), (10, 100)]
    best = read_checkpoint(branch / 'best.pt')
    assert best['best_metric'] == 10. and best['iteration'] == 1
    assert best['config']['mcts_sims'] == 100


def test_auto_workers_reserve_core_limit_games_and_allow_override(monkeypatch):
    import common.parallel as parallel
    monkeypatch.setattr(parallel, 'available_cpu_cores', lambda: 12)
    assert parallel.resolve_workers(None, 8) == 8
    assert parallel.resolve_workers(None, 100) == 11
    assert parallel.resolve_workers(None, 1) == 1
    assert parallel.resolve_workers(4, 8) == 4
    assert parallel.resolve_workers(1, 8) == 1
    monkeypatch.setattr(parallel, 'available_cpu_cores', lambda: 1)
    assert parallel.resolve_workers(None, 8) == 1
    with pytest.raises(ValueError):
        parallel.resolve_workers(0, 8)
    with pytest.raises(ValueError):
        parallel.resolve_workers(None, 0)


def test_cpu_detection_respects_affinity_and_mac_performance_cores(monkeypatch):
    import common.parallel as parallel
    monkeypatch.setattr(parallel.os, 'process_cpu_count', lambda: 16, raising=False)
    monkeypatch.setattr(parallel.os, 'sched_getaffinity', lambda _: set(range(6)), raising=False)
    monkeypatch.setattr(parallel.sys, 'platform', 'darwin')
    monkeypatch.setattr(parallel.subprocess, 'check_output', lambda *a, **kw: '12\n')
    assert parallel.available_cpu_cores() == 6
    monkeypatch.setattr(parallel.os, 'sched_getaffinity', lambda _: set(range(16)))
    assert parallel.available_cpu_cores() == 12
    def missing_sysctl(*args, **kwargs):
        raise OSError('not available')
    monkeypatch.setattr(parallel.subprocess, 'check_output', missing_sysctl)
    assert parallel.available_cpu_cores() == 16
    monkeypatch.setattr(parallel.os, 'process_cpu_count', lambda: None)
    assert parallel.available_cpu_cores() == 1


def test_resume_uses_current_auto_workers_instead_of_saved_count(tmp_path, monkeypatch):
    import common.parallel as parallel
    from common.training import resolve_args, training_parser
    monkeypatch.setattr(parallel, 'available_cpu_cores', lambda: 12)
    model = ActorCritic()
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 'a2c',
                    dict(workers=4, episodes_per_update=8), 0.)
    defaults = dict(episodes_per_update=8)
    args = resolve_args(training_parser('test'), 'a2c', defaults,
                        ['--iterations', '2', '--resume', str(path)])
    assert args.workers == 8 and args.workers_auto
    args = resolve_args(training_parser('test'), 'a2c', defaults,
                        ['--iterations', '2', '--resume', str(path), '--workers', '2'])
    assert args.workers == 2 and not args.workers_auto


def test_auto_workers_training_and_resume_match(tmp_path, monkeypatch):
    import common.parallel as parallel
    import plot
    monkeypatch.setattr(parallel, 'available_cpu_cores', lambda: 3)
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--episodes-per-update', '4', '--eval-episodes', '2', '--eval-every', '1']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    a2c.main(['--iterations', '2', '--save-dir', str(whole), *options])
    a2c.main(['--iterations', '1', '--save-dir', str(split), *options])
    a2c.main(['--iterations', '2', '--resume', str(split / 'last.pt')])
    expected, actual = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert actual['config']['workers'] == 2 and actual['config']['workers_auto']
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)
