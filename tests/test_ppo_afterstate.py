"""Afterstate causality, unchanged game rules, PPO targets and run continuity."""
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from algorithms import ppo_afterstate
from algorithms.ppo_afterstate import (AfterstateActorCritic, collect_afterstate,
                                      decision_values, encode_candidates, evaluate_afterstate,
                                      masked_policy_kl, update)
from board import Board
from common.checkpoints import (export_model, model_from_checkpoint, read_checkpoint,
                                save_checkpoint)
from common.models import ActorCritic, masked_categorical
from common.parallel import GamePool
from common.rollout import Rollout
from common.training import setup
from gym2048_afterstate_env import (Gym2048AfterstateEnv, candidate_afterstates,
                                   move_afterstate, spawn_outcomes)
from gym2048_env import Gym2048Env


class NoSpawnBoard(Board):
    def add_random_tile(self):
        pass


@pytest.mark.parametrize('size', [2, 3, 4, 5])
def test_candidates_match_original_deterministic_moves_without_mutation(size):
    rng = np.random.default_rng(921)
    for _ in range(30):
        matrix = rng.choice([0, 0, 2, 4, 8, 16, 32768], (size, size))
        original = matrix.copy()
        rng_state = random.getstate()
        candidates, mask, merges = candidate_afterstates(matrix)
        assert random.getstate() == rng_state
        np.testing.assert_array_equal(matrix, original)
        assert mask.tolist() == Board(matrix, size=size).can_move_dir
        for action in range(4):
            reference = NoSpawnBoard(matrix, size=size)
            reference.move(action)
            np.testing.assert_array_equal(candidates[action], reference.matrix)
            assert merges[action] == reference.merge_reward
            assert candidates[action].sum() == matrix.sum()
    with pytest.raises(ValueError, match='Invalid action'):
        move_afterstate(matrix, 4)


def test_split_phases_do_not_draw_rng_until_spawn_and_enforce_order():
    matrix = np.array([[2, 2, 2, 2], *[[0] * 4] * 3])
    env = Gym2048AfterstateEnv(matrix=matrix, seed=73)
    original = Gym2048Env(matrix=matrix, seed=73)
    before_rng = env._rng.getstate()
    candidates, mask, _ = env.afterstates()
    afterstate, reward, done, truncated, info = env.step_move(0)
    assert env._rng.getstate() == before_rng
    np.testing.assert_array_equal(afterstate, candidates[0])
    np.testing.assert_array_equal(afterstate[0], [4, 4, 0, 0])
    assert reward == 0 and not done and not truncated
    assert info['merge_reward'] == 8 and info['phase'] == 'afterstate'
    assert env.legal_actions() == []
    assert mask[0]
    for action in (lambda: env.step_move(0), lambda: env.step(0), env.afterstates):
        with pytest.raises(RuntimeError, match='step_spawn'):
            action()
    clone = env.clone()
    state, reward, done, truncated, info = env.step_spawn()
    copied = clone.step_spawn()
    expected = original.step(0)
    np.testing.assert_array_equal(state, copied[0])
    np.testing.assert_array_equal(state, expected[0])
    assert (reward, done, truncated) == expected[1:4]
    assert env._rng.getstate() == original._rng.getstate()
    row, column = info['spawn_position']
    assert afterstate[row, column] == 0
    assert state[row, column] == info['spawn_value'] == reward
    assert np.count_nonzero(state != afterstate) == 1
    assert info['phase'] == 'decision'
    with pytest.raises(RuntimeError, match='preceding legal'):
        env.step_spawn()


def test_complete_games_match_original_seeds_rewards_and_statistics():
    for seed in range(8):
        original, split = Gym2048Env(), Gym2048AfterstateEnv()
        obs, info = original.reset(seed=seed)
        other, other_info = split.reset(seed=seed)
        np.testing.assert_array_equal(obs, other)
        assert np.count_nonzero(other) == 1
        actions = random.Random(seed + 51)
        for _ in range(3000):
            # Include illegal moves, and enumerate unused candidates each turn.
            before_rng = split._rng.getstate()
            split.afterstates()
            assert split._rng.getstate() == before_rng
            action = actions.randrange(4)
            expected = original.step(action)
            actual = split.step(action)
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
            assert all(expected[4][key] == actual[4][key] for key in expected[4])
            assert original._rng.getstate() == split._rng.getstate()
            if expected[2]:
                break
        else:
            pytest.fail('Game should terminate')


def test_termination_occurs_after_chance_and_illegal_moves_never_spawn():
    matrix = [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [0, 32, 64, 128]]
    env = Gym2048AfterstateEnv(matrix=matrix, seed=15)
    afterstate, reward, done, _, _ = env.step_move(0)
    assert not done and reward == 0 and np.count_nonzero(afterstate == 0) == 1
    state, reward, done, _, info = env.step_spawn()
    assert done and not any(info['can_move_dir']) and reward in (2, 4)
    before_rng = env._rng.getstate()
    other, reward, done, _, info = env.step(0)
    np.testing.assert_array_equal(other, state)
    assert reward == 0 and done and info['afterstate'] is None
    assert env._rng.getstate() == before_rng


def test_chance_outcomes_have_exact_cell_and_tile_probabilities():
    matrix = np.full((4, 4), 8)
    matrix[0, 1] = matrix[3, 2] = 0
    before = matrix.copy()
    rng_state = random.getstate()
    boards, probabilities, rewards = spawn_outcomes(matrix)
    np.testing.assert_allclose(probabilities, [.45, .05, .45, .05])
    assert probabilities.sum() == pytest.approx(1.)
    assert probabilities @ rewards == pytest.approx(2.2)
    assert np.all(np.count_nonzero(boards != matrix, axis=(1, 2)) == 1)
    np.testing.assert_array_equal(matrix, before)
    assert random.getstate() == rng_state
    with pytest.raises(ValueError, match='empty cell'):
        spawn_outcomes(np.ones((4, 4)))


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_policy_scores_shared_afterstates_masks_and_learns(architecture):
    setup(19)
    model = AfterstateActorCritic(architecture=architecture)
    states = [np.array([[2, 2, 4, 0], *[[0] * 4] * 3]),
              np.array([[2, 0, 0, 0], *[[0] * 4] * 3])]
    candidates, masks = encode_candidates(states)
    logits, values = model(candidates)
    assert logits.shape == values.shape == (2, 4)
    torch.testing.assert_close(model(candidates[0])[0], logits[0], atol=1e-6, rtol=1e-5)
    permutation = torch.tensor([3, 0, 2, 1])
    # A shared scalar scorer has no direction-specific hidden policy head.
    permuted_logits, permuted_values = model(candidates[:, permutation])
    torch.testing.assert_close(permuted_logits, logits[:, permutation], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(permuted_values, values[:, permutation], atol=1e-6, rtol=1e-5)
    distribution = masked_categorical(logits, masks)
    assert torch.all(distribution.probs[~masks] == 0)
    chosen = distribution.probs.argmax(-1)
    loss = -distribution.log_prob(chosen).mean() + (values.gather(1, chosen[:, None]) - 1).square().mean()
    loss.backward()
    for parameter in (model.embedding.weight, model.policy_head.weight, model.value_head.weight):
        assert parameter.grad.abs().sum() > 0


def test_decision_baseline_is_policy_weighted_afterstate_value():
    logits = torch.zeros(4, requires_grad=True)
    values = torch.tensor([2., 1000., 6., 1000.], requires_grad=True)
    distribution = masked_categorical(logits, [True, False, True, False])
    baseline = decision_values(distribution, values)
    assert baseline.item() == 4
    baseline.backward()
    assert logits.grad is None
    torch.testing.assert_close(values.grad, torch.tensor([.5, 0., .5, 0.]))


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_real_rollout_keeps_ppo_ratio_and_one_discount_per_full_move(architecture):
    setup(11)
    model = AfterstateActorCritic(architecture=architecture)
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.zero_()
    rollout = collect_afterstate(Gym2048AfterstateEnv(), model, 1, .9, 845, td_steps=1)
    logits, _ = model(rollout.states)
    distribution = masked_categorical(logits, rollout.masks)
    ratios = (distribution.log_prob(rollout.actions) - rollout.old_log_probs).exp()
    torch.testing.assert_close(ratios, torch.ones_like(ratios), atol=1e-6, rtol=1e-6)
    # Zero frozen V: every one-step target is just the current spawn/128.
    assert set(rollout.returns.tolist()) <= {2/128, 4/128}
    assert rollout.returns.sum().item() * 128 == rollout.episodes[0]['spawn_return']
    before_policy = model.policy_head.weight.detach().clone()
    before_value = model.value_head.weight.detach().clone()
    metrics = update(model, torch.optim.Adam(model.parameters(), lr=3e-4), rollout, epochs=1)
    assert metrics['updates'] > 0
    assert not torch.equal(before_policy, model.policy_head.weight)
    assert not torch.equal(before_value, model.value_head.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_returns_use_next_decision_baseline_and_zero_terminal_bootstrap():
    setup(17)
    model = AfterstateActorCritic(architecture='cnn2x2')
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.fill_(3.)
    rollout = collect_afterstate(Gym2048AfterstateEnv(), model, 2, .9, 92, td_steps=1)
    offset = 0
    for episode in rollout.episodes:
        targets = rollout.returns[offset:offset + episode['steps']]
        assert torch.all((targets[:-1] >= 2/128 + .9*3 - 1e-6)
                         & (targets[:-1] <= 4/128 + .9*3 + 1e-6))
        assert targets[-1].item() in (2/128, 4/128)
        offset += episode['steps']


def test_critic_trains_only_the_selected_afterstate_and_kl_can_stop_updates():
    class FixedCandidates(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(4))
            self.values = nn.Parameter(torch.tensor([1., 2., 3., 4.]))

        def forward(self, candidates):
            return self.logits.expand(len(candidates), -1), self.values.expand(len(candidates), -1)

    model = FixedCandidates()
    candidates = torch.zeros(2, 4, 16)
    masks = torch.ones(2, 4, dtype=torch.bool)
    actions = torch.tensor([0, 3])
    distribution = masked_categorical(model(candidates)[0], masks)
    rollout = Rollout(candidates, masks, actions, distribution.log_prob(actions).detach(),
                      distribution.logits.detach(), torch.zeros(2), torch.tensor([2., 6.]), [])
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    metrics = update(model, optimizer, rollout, epochs=1, entropy_coef=0)
    assert metrics['critic'] == F.smooth_l1_loss(torch.tensor([1., 4.]), rollout.returns).item()
    assert model.values.grad[1:3].abs().sum() == 0
    assert model.values.grad[[0, 3]].abs().sum() > 0
    with torch.no_grad():
        model.logits[0] = 4
    before = model.values.clone()
    metrics = update(model, optimizer, rollout, target_kl=.0001)
    assert metrics['updates'] == 0 and metrics['kl'] > .0001
    torch.testing.assert_close(before, model.values)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_log_prob_kl_survives_legal_probability_underflow(device):
    mask = torch.tensor([[True, True, True, False]], device=device)
    old_logits = torch.tensor([[0., -86.5, -2., 5000.]], device=device)
    new_logits = torch.tensor([[.03, -120., -2.3, -5000.]], device=device)
    old, new = masked_categorical(old_logits, mask), masked_categorical(new_logits, mask)
    assert old.probs[0, 1] > 0 and new.probs[0, 1] == 0
    assert torch.isinf(torch.distributions.kl_divergence(old, new)).all()
    actual = masked_policy_kl(old.logits, new.logits, mask)
    reference_old = masked_categorical(old_logits.cpu().double(), mask.cpu())
    reference_new = masked_categorical(new_logits.cpu().double(), mask.cpu())
    expected = torch.distributions.kl_divergence(reference_old, reference_new)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-7, rtol=2e-5)
    assert 0 < actual.item() < .02
    assert masked_policy_kl(old.logits, old.logits, mask).item() == 0


def test_log_prob_kl_matches_regular_kl_and_ignores_illegal_infinities():
    setup(93)
    mask = torch.rand(17, 4) > .5
    mask[:, 0] = True
    old = masked_categorical(torch.randn(17, 4, dtype=torch.float64), mask)
    new = masked_categorical(torch.randn(17, 4, dtype=torch.float64), mask)
    expected = torch.distributions.kl_divergence(old, new)
    old_logs, new_logs = old.logits.clone(), new.logits.clone()
    old_logs[~mask] = -torch.inf
    new_logs[~mask] = torch.nan
    torch.testing.assert_close(masked_policy_kl(old_logs, new_logs, mask), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_underflow_does_not_stop_ppo_but_real_kl_still_does(device):
    import json

    class FixedCandidates(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([.03, -120., -2.3, -5000.]))
            self.values = nn.Parameter(torch.zeros(4))

        def forward(self, candidates):
            return self.logits.expand(len(candidates), -1), self.values.expand(len(candidates), -1)

    model = FixedCandidates().to(device)
    candidates = torch.zeros(5, 4, 16, device=device)
    masks = torch.tensor([[True, True, True, False]] * 5, device=device)
    actions = torch.tensor([0, 2, 0, 0, 2], device=device)
    old = masked_categorical(torch.tensor([0., -86.5, -2., 5000.], device=device).expand(5, -1), masks)
    rollout = Rollout(candidates, masks, actions, old.log_prob(actions), old.logits,
                      torch.arange(5, dtype=torch.float32, device=device),
                      torch.full((5,), .5, device=device), [])
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    metrics = update(model, optimizer, rollout, epochs=3, batch_size=2)
    assert metrics['updates'] == metrics['planned_updates'] == metrics['kl_checks'] == 9
    assert metrics['samples_updated'] == 15  # Include the one-sample tail in every epoch.
    assert metrics['effective_epochs'] == 3 and metrics['update_fraction'] == 1
    assert metrics['update_stop_reason'] == 'epochs_complete'
    assert metrics['kl_method'] == ppo_afterstate.KL_METHOD
    assert metrics['kl_nonfinite_checks'] == 0
    assert 0 < metrics['rollout_kl'] < .02
    assert metrics['zero_legal_probabilities'] == 5
    assert metrics['legal_logit_span_max'] == pytest.approx(120.03, abs=1e-4)
    assert metrics['critic_mean'] == pytest.approx(.125)
    json.dumps(metrics, allow_nan=False)
    with torch.no_grad():
        model.logits[0] = 4
    stopped = update(model, optimizer, rollout, epochs=3, batch_size=2)
    assert stopped['updates'] == stopped['samples_updated'] == stopped['effective_epochs'] == 0
    assert stopped['update_stop_reason'] == 'target_kl'
    assert stopped['kl'] > .02 and stopped['rollout_kl'] > .02


def test_rollout_diagnostics_use_all_samples_with_final_policy():
    class IndexedCandidates(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.],
                                                     [2., 0., 0., 0.], [3., 0., 0., 0.],
                                                     [4., 0., 0., 0.]]))
            self.values = nn.Parameter(torch.zeros(5, 4))

        def forward(self, candidates):
            ids = candidates[:, 0, 0].long()
            return self.logits[ids], self.values[ids]

    model = IndexedCandidates()
    candidates = torch.zeros(5, 4, 16)
    candidates[:, 0, 0] = torch.arange(5)
    masks = torch.ones(5, 4, dtype=torch.bool)
    actions = torch.zeros(5, dtype=torch.long)
    old = masked_categorical(torch.zeros(5, 4), masks)
    rollout = Rollout(candidates, masks, actions, old.log_prob(actions), old.logits,
                      torch.zeros(5), torch.zeros(5), [])
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    metrics = update(model, optimizer, rollout, batch_size=2, target_kl=1e-5)
    final = masked_categorical(model.logits, masks)
    assert metrics['updates'] == 0
    assert metrics['rollout_kl'] == pytest.approx(float(torch.distributions.kl_divergence(old, final).mean().detach()), abs=1e-6)
    assert metrics['rollout_entropy'] == pytest.approx(float(final.entropy().mean().detach()), abs=1e-6)


def test_parallel_collection_and_evaluation_match_serial():
    setup(31)
    model = AfterstateActorCritic(architecture='cnn2x2')
    serial = collect_afterstate(Gym2048AfterstateEnv(), model, 2, .999, 654)
    expected = evaluate_afterstate(model, 2, 655)
    with GamePool(2) as pool:
        parallel = collect_afterstate(Gym2048AfterstateEnv(), model, 2, .999, 654, pool=pool)
        actual = evaluate_afterstate(model, 2, 655, pool)
    assert serial.episodes == parallel.episodes
    for field in ('states', 'masks', 'actions', 'old_log_probs', 'old_logits', 'advantages', 'returns'):
        torch.testing.assert_close(getattr(serial, field), getattr(parallel, field), atol=2e-6, rtol=2e-6)
    assert expected['results'] == actual['results']
    assert model.training


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_checkpoint_export_play_and_evaluation_compatibility(tmp_path, architecture):
    from common.evaluation import evaluate
    from play import make_agent
    setup(4)
    model = AfterstateActorCritic(architecture=architecture)
    full, exported = tmp_path / 'last.pt', tmp_path / 'export.pt'
    save_checkpoint(full, model, torch.optim.Adam(model.parameters()), 1, 'ppo_afterstate', {}, 0)
    restored = model_from_checkpoint(read_checkpoint(full))
    state, info = Gym2048AfterstateEnv().reset(seed=77)
    candidates, masks = encode_candidates([state])
    torch.testing.assert_close(model(candidates)[0], restored(candidates)[0], atol=0, rtol=0)
    export_model(full, exported)
    restored = model_from_checkpoint(torch.load(exported, weights_only=True))
    torch.testing.assert_close(model(candidates)[1], restored(candidates)[1], atol=0, rtol=0)
    action, _, metadata = make_agent('ppo_afterstate', exported)
    assert masks[0, action(state, info)]
    assert metadata['transition_model'] == 'exact_move_then_spawn'
    # The existing evaluator/play environment has identical full-step rules.
    expected = evaluate_afterstate(model, 1, 66)
    assert evaluate(model, 1, 66)['results'] == expected['results']
    actual = evaluate(None, 1, 66, lambda state, info, _: action(state, info))
    assert expected['results'] == actual['results']
    with pytest.raises(ValueError, match='full training checkpoint'):
        ppo_afterstate.main(['--iterations', '2', '--resume', str(exported)])


@pytest.mark.parametrize('architecture,workers', [('vit', '1'), ('cnn2x2', '2')])
def test_training_resume_matches_uninterrupted_and_preserves_configuration(tmp_path, monkeypatch,
                                                                         architecture, workers):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    options = ['--device', 'cpu', '--workers', workers, '--architecture', architecture,
               '--episodes-per-update', '2', '--eval-episodes', '1', '--eval-every', '1', '--seed', '7']
    ppo_afterstate.main(['--iterations', '2', '--save-dir', str(whole), *options])
    ppo_afterstate.main(['--iterations', '1', '--save-dir', str(split), *options])
    ppo_afterstate.main(['--iterations', '2', '--resume', str(split / 'last.pt'), '--workers', workers])
    expected, actual = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert actual['algorithm'] == 'ppo_afterstate'
    assert actual['model_config']['model_type'] == 'ppo_afterstate'
    assert actual['config']['kl_method'] == ppo_afterstate.KL_METHOD
    assert actual['iteration'] == 2
    for key in ppo_afterstate.DEFAULTS:
        if key not in ('architecture', 'episodes_per_update'):
            assert actual['config'][key] == ppo_afterstate.DEFAULTS[key]
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)
    with pytest.raises(SystemExit):
        ppo_afterstate.main(['--iterations', '3', '--resume', str(split / 'last.pt'), '--gamma', '.95'])


def test_default_uses_best_ppo_cnn_configuration_and_old_ppo_cannot_resume(tmp_path):
    assert AfterstateActorCritic().architecture == 'cnn2x2'
    assert ppo_afterstate.DEFAULTS['gamma'] == .999
    assert ppo_afterstate.DEFAULTS['td_steps'] == 10
    assert ppo_afterstate.DEFAULTS['td_lambda'] == .5
    setup(4)
    model = ActorCritic(architecture='vit')
    path = tmp_path / 'ppo.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 'ppo', {}, 0)
    with pytest.raises(SystemExit):
        ppo_afterstate.main(['--iterations', '2', '--resume', str(path)])
