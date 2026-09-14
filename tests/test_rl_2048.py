import random
import numpy as np
import pytest
import torch

from board import Board
from gym2048_env import Gym2048Env
from common.models import ActorCritic, masked_categorical, preprocess_observation
from common.rollout import compute_gae, collect_actor_critic, discounted_returns
from common.evaluation import evaluate
from common.checkpoints import save_checkpoint, load_checkpoint
from common.training import setup
from algorithms.ppo import clipped_surrogate, update as update_ppo
from algorithms.a2c import update as update_a2c
from algorithms.alphazero import (
    MCTS, Node, PolicyValueNet, softmax_visit_probs, self_play_game,
    train_policy_value, augment_board_and_policy,
)
from algorithms.mcts import RolloutMCTS



@pytest.fixture(autouse=True)
def deterministic():
    setup(42)


def test_a2c_actor_and_entropy_reach_policy_head():
    model = ActorCritic()
    logits, _ = model(torch.arange(16).float())
    dist = masked_categorical(logits, [True, False, True, False])
    action = dist.sample()
    logp = dist.log_prob(action)
    (-logp).backward()
    assert model.policy_head.weight.grad.abs().sum() > 0
    logits = torch.tensor([1.2, 99., -0.5, 88.], requires_grad=True)
    dist = masked_categorical(logits, [True, False, True, False])
    dist.entropy().backward()
    assert logits.grad[[0, 2]].abs().sum() > 0
    assert torch.equal(logits.grad[[1, 3]], torch.zeros(2))
    assert torch.equal(dist.probs[[1, 3]], torch.zeros(2))


def test_mask_batch_single_action_and_terminal():
    logits = torch.randn(2, 4, requires_grad=True)
    mask = torch.tensor([[True, False, False, False], [False, True, False, True]])
    dist = masked_categorical(logits, mask)
    assert dist.probs[0, 0] == 1
    assert dist.entropy()[0] == 0
    for _ in range(20):
        actions = dist.sample()
        assert mask[torch.arange(2), actions].all()
    with pytest.raises(ValueError):
        masked_categorical(logits, torch.zeros_like(mask))


def test_unchanged_ppo_ratio_is_one_with_original_masks():
    model = ActorCritic()
    rollout = collect_actor_critic(Gym2048Env(), model, 2, .997, 123)
    dist = masked_categorical(model(rollout.states)[0], rollout.masks)
    ratio = (dist.log_prob(rollout.actions) - rollout.old_log_probs).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)
    assert not rollout.old_log_probs.requires_grad
    assert not rollout.returns.requires_grad


def test_clipping_keeps_corrective_gradients_for_both_advantage_signs():
    ratio = torch.tensor([.5, 1.5, 1.5, .5], requires_grad=True)
    advantages = torch.tensor([1., -1., 1., -1.])
    loss = -clipped_surrogate(ratio, advantages, .2).sum()
    loss.backward()
    torch.testing.assert_close(ratio.grad, torch.tensor([-1., 1., 0., 0.]))


def test_gae_terminal_vs_truncation_and_no_cross_episode_leak():
    adv, ret = compute_gae(torch.tensor([1., 2., 100.]), torch.tensor([.5, .7, .9]),
                          torch.tensor([.7, 10., 20.]), torch.tensor([False, True, False]),
                          torch.tensor([False, True, True]), gamma=.9, gae_lambda=1.)
    torch.testing.assert_close(ret, torch.tensor([2.8, 2., 118.]))
    torch.testing.assert_close(adv, ret - torch.tensor([.5, .7, .9]))


@pytest.mark.parametrize('update', [update_a2c, update_ppo])
def test_real_update_changes_actor_weights(update):
    model = ActorCritic()
    rollout = collect_actor_critic(Gym2048Env(), model, 2, .997, 345)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = model.policy_head.weight.detach().clone()
    metrics = update(model, optimizer, rollout)
    assert metrics['updates'] >= 1
    assert not torch.equal(before, model.policy_head.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_kl_guard_skips_stale_update():
    model = ActorCritic()
    rollout = collect_actor_critic(Gym2048Env(), model, 1, .997, 8)
    with torch.no_grad():
        model.policy_head.bias[0] += 10
    optimizer = torch.optim.Adam(model.parameters())
    before = model.policy_head.weight.detach().clone()
    metrics = update_ppo(model, optimizer, rollout, target_kl=.0001)
    assert metrics['updates'] == 0
    assert torch.equal(before, model.policy_head.weight)


def test_merge_reward_and_rng_isolation():
    matrix = np.array([[2, 2, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    a, b = Gym2048Env(matrix=matrix, seed=10), Gym2048Env(matrix=matrix, seed=10)
    before_random = random.getstate()
    before_numpy = np.random.get_state()
    other = Gym2048Env(seed=900)
    other.reset(seed=123)
    assert random.getstate() == before_random
    np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
    clone = a.clone()
    clone.step(0)  # Cloning and simulation do not change the original's spawn stream.
    state, reward, _, _, info = a.step(0)
    state_b, _, _, _, _ = b.step(0)
    np.testing.assert_array_equal(state, state_b)
    assert reward in (2, 4)  # legacy reward is unchanged
    assert info['merge_reward'] == 12
    assert info['merge_score'] == 12
    assert info['score'] == state.sum() / 500
    _, _, _, _, info = a.step(0)
    assert info['merge_score'] >= 12


def test_all_eight_symmetries_preserve_legal_actions_and_afterstates():
    matrix = np.array([[2, 2, 0, 4], [8, 4, 4, 0], [2, 0, 2, 0], [16, 8, 0, 2]])
    pi = np.arange(4, dtype=np.float32) + 1
    pairs = augment_board_and_policy(matrix, pi)
    assert len(pairs) == 8
    for action in range(4):
        board = Board(matrix)
        # Deterministic afterstate: suppress stochastic tile addition.
        board.add_random_tile = lambda: None
        board.move(action)
        outcomes = augment_board_and_policy(board.matrix, pi)
        for (transformed, transformed_pi), (expected, _) in zip(pairs, outcomes):
            mapped_action = int(np.flatnonzero(transformed_pi == pi[action])[0])
            candidate = Board(transformed)
            candidate.add_random_tile = lambda: None
            candidate.move(mapped_action)
            np.testing.assert_array_equal(candidate.matrix, expected)
            assert candidate.merge_reward == board.merge_reward


def test_alphazero_chance_resampling_and_modes():
    model = PolicyValueNet()
    matrix = np.zeros((4, 4), dtype=int); matrix[0, 0] = 2
    root = Node(1., matrix=matrix)
    search = MCTS(model, torch.device('cpu'), dirichlet_frac=0, seed=3)
    model.train()
    search.run(root, 80)
    assert model.training
    assert sum(e.visit_count for e in root.children.values()) == 80
    assert any(len(e.outcomes) > 1 for e in root.children.values())
    assert set(root.children) == {2, 3}
    assert sum(e.prior for e in root.children.values()) == pytest.approx(1.)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_alphazero_backup_uses_spawn_not_merge_reward():
    model = PolicyValueNet()
    for parameter in model.parameters():
        parameter.data.zero_()
    matrix = np.array([[512, 512, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    root = Node(1., matrix=matrix, legal_mask=np.array([True, False, False, False]))
    MCTS(model, torch.device('cpu'), dirichlet_frac=0).run(root, 1)
    assert root.children[0].q_value in (2 / 128, 4 / 128)


def test_visit_temperature_power_law_and_missing_actions():
    children = {0: Node(.4, visit_count=10), 2: Node(.6, visit_count=20)}
    np.testing.assert_allclose(softmax_visit_probs(children, 1), [1 / 3, 0, 2 / 3, 0])
    np.testing.assert_allclose(softmax_visit_probs(children, .5), [.2, 0, .8, 0])
    np.testing.assert_array_equal(softmax_visit_probs(children, 0), [0, 0, 1, 0])
    np.testing.assert_allclose(softmax_visit_probs({1: Node(.2), 3: Node(.8)}, 1), [0, .2, 0, .8])


def test_alphazero_selfplay_train_and_terminal_return():
    model = PolicyValueNet()
    examples, info = self_play_game(model, torch.device('cpu'), 2, 10, seed=123)
    assert len(examples) == info['steps']
    assert len(set(ex.value for ex in examples)) > 1
    assert examples[-1].value in (2 / 128, 4 / 128)
    assert examples[0].value == pytest.approx(info['spawn_return'] / 128)
    for ex in examples:
        assert ex.policy.sum() == pytest.approx(1)
        assert not ex.policy[~np.array(Board(ex.state).can_move_dir)].any()
    before = model.policy_head.weight.detach().clone()
    pl, vl = train_policy_value(model, torch.optim.Adam(model.parameters()), examples[:32], torch.device('cpu'))
    assert np.isfinite(pl + vl)
    assert not torch.equal(before, model.policy_head.weight)


def test_evaluation_uses_all_seeds_and_restores_model_mode():
    model = ActorCritic().train()
    first = evaluate(model, episodes=3, seed=999)
    second = evaluate(model, episodes=3, seed=999)
    assert len(first['results']) == 3
    assert first['results'] == second['results']
    assert model.training
    assert first['p4096'] <= first['p2048'] <= first['p1024'] <= first['p512']


def test_checkpoint_optimizer_rng_roundtrip(tmp_path):
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, optimizer, 3, 'ppo', {'gamma': .997}, 100)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    with torch.no_grad():
        model.policy_head.bias.add_(10)
    data = load_checkpoint(path, model, optimizer, restore_rng=True)
    assert data['iteration'] == 3
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])
    torch.testing.assert_close(model.policy_head.bias, torch.zeros(4))


def test_demo_teacher_deterministic_and_does_not_consume_global_rng():
    matrix = np.zeros((4, 4), dtype=int); matrix[0, 0] = 2
    rng = random.getstate()
    first = RolloutMCTS(2, 123).search(matrix)
    second = RolloutMCTS(2, 123).search(matrix)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert random.getstate() == rng


def test_resumed_update_matches_uninterrupted_training(tmp_path):
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters())
    rollout = collect_actor_critic(Gym2048Env(), model, 1, .997, 420)
    update_ppo(model, optimizer, rollout, epochs=1)
    checkpoint = tmp_path / 'resume.pt'
    save_checkpoint(checkpoint, model, optimizer, 1, 'ppo', {}, 0)
    rollout_next = collect_actor_critic(Gym2048Env(), model, 1, .997, 421)
    update_ppo(model, optimizer, rollout_next, epochs=2)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = ActorCritic()
    restored_optimizer = torch.optim.Adam(restored.parameters())
    load_checkpoint(checkpoint, restored, restored_optimizer, restore_rng=True)
    resumed_rollout = collect_actor_critic(Gym2048Env(), restored, 1, .997, 421)
    torch.testing.assert_close(rollout_next.actions, resumed_rollout.actions)
    update_ppo(restored, restored_optimizer, resumed_rollout, epochs=2)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


def test_mass_conservation_and_evaluation_objective():
    # Merging changes representation but never creates tile mass.
    env = Gym2048Env(matrix=[[2048, 1024, 512, 0], [0]*4, [0]*4, [0]*4], seed=22)
    initial_mass = env.board.matrix.sum()
    cumulative = 0
    for _ in range(40):
        state, reward, done, _, info = env.step(env.legal_actions()[0])
        cumulative += reward
        assert state.sum() == initial_mass + cumulative
        if done:
            break
    model = ActorCritic()
    metrics = evaluate(model, episodes=3, seed=333)
    assert metrics['score_metric'] == 'spawn_mass'
    assert metrics['mean_return'] == np.mean([r['spawn_return'] for r in metrics['results']])
    assert metrics['mean_score'] == metrics['mean_return']
    for row in metrics['results']:
        assert row['board_sum'] - row['spawn_return'] in (2, 4)
        assert 2 * row['steps'] <= row['spawn_return'] <= 4 * row['steps']


def test_gae_collector_uses_spawn_rewards_even_with_large_merges():
    class TwoStepEnv:
        def reset(self, seed=None):
            self.t = 0
            return np.zeros((4, 4)), {'can_move_dir': [True] * 4}

        def step(self, action):
            self.t += 1
            return np.zeros((4, 4)), 2 * self.t, self.t == 2, False, {
                'can_move_dir': [True] * 4, 'merge_reward': 4096,
                'merge_score': 8192, 'max_value': 4096}

    model = ActorCritic()
    rollout = collect_actor_critic(TwoStepEnv(), model, 1, gamma=1., seed=0, gae_lambda=1.)
    torch.testing.assert_close(rollout.returns, torch.tensor([6 / 128, 4 / 128]))
    assert rollout.episodes[0]['spawn_return'] == 6


def test_checkpoint_rejects_different_reward_objective(tmp_path):
    model = ActorCritic()
    path = tmp_path / 'wrong_objective.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 'ppo', {}, 0)
    data = torch.load(path, weights_only=False)
    data['reward_objective'] = 'merge_points'
    torch.save(data, path)
    with pytest.raises(ValueError, match='reward objective'):
        load_checkpoint(path, model)
