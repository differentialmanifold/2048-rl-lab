"""Frozen-world afterstate PPO contracts, including uninterrupted/resumed runs.

Passing world reports in fixtures test provenance, not model accuracy.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from algorithms.latent_imagination import train as trainer
from algorithms.latent_imagination.policy import ImaginationAgent
from algorithms.latent_imagination.rollout import collect_imagined, collect_games, pack_records
from algorithms.latent_imagination.world.model import WorldModel
from algorithms.latent_imagination.world.dynamics import world_fingerprint
from algorithms.latent_imagination.world.data import generate, SyntheticData, atomic_json
from algorithms.latent_imagination.world.boundaries import prepare_boundaries, CurriculumData
from algorithms.latent_imagination.runtime import file_digest
from algorithms.ppo_afterstate import update, DEFAULTS as REFERENCE_DEFAULTS
from common.checkpoints import read_checkpoint, save_checkpoint, model_from_checkpoint, export_model
from common.parallel import GamePool
from common.training import setup


def tiny_world():
    model = WorldModel(model_version=2, width=16, latent_dim=4,
        tokenizer_layers=1, dynamics_layers=1, heads=2, state_head_version=2)
    with torch.no_grad():
        model.world.terminal[-1].weight.zero_();model.world.terminal[-1].bias.fill_(-10)
        model.world.legality[-1].weight.zero_();model.world.legality[-1].bias.fill_(10)
        model.world.reward[-1].weight.zero_();model.world.reward[-1].bias.copy_(torch.tensor([-10.,10.,-10.]))
        model.world.action_transition.output.weight.normal_(std=.01)
    return model


def tiny_model(architecture='vit'):
    source = tiny_world()
    model = ImaginationAgent(source.model_config, architecture)
    model.world.load_state_dict(source.world.state_dict())
    return model


@pytest.fixture(scope='module')
def verified_fixture(tmp_path_factory):
    setup(12, 'cpu')
    root = tmp_path_factory.mktemp('verified-latent-ppo')
    generate(root / 'data', 180, 10, 17, 1, 6)
    base = SyntheticData(root / 'data')
    owners = prepare_boundaries(root / 'head_data', base, 180, 17, 6, 1)
    data = CurriculumData(base, root / 'head_data', owners)
    source = tiny_world()
    (root / 'world10').mkdir();(root / 'audit').mkdir()
    checkpoint = root / 'world10/passed.pt'
    save_checkpoint(checkpoint, source, torch.optim.Adam(source.parameters()), 10, 'latent_dreamer', {}, 0.)
    report = dict(passed=True, consecutive_passes=2, iteration=10, data_sha256=data.digest,
                  checkpoint_sha256=file_digest(checkpoint), world_sha256=world_fingerprint(source.world),
                  split='validation', games=[dict(passed=True)])
    atomic_json(root / 'world10/validation.json', report)
    atomic_json(root / 'audit/validation.json', report)
    return root


def test_shared_spatial_vit_and_candidate_order_equivariance():
    from algorithms.latent_imagination.transformer import SwiGLU
    setup(1, 'cpu');model = tiny_model()
    assert len(model.policy.blocks) == 2
    assert all(isinstance(block.mlp, SwiGLU) and block.heads == 4 for block in model.policy.blocks)
    assert not any(isinstance(module, torch.nn.Conv2d) for module in model.policy.modules())
    candidates = torch.randn(3, 4, 16, 4)
    logits, values = model.policy(candidates)
    order = [2, 0, 3, 1]
    a, b = model.policy(candidates[:, order])
    torch.testing.assert_close(a, logits[:, order]);torch.testing.assert_close(b, values[:, order])
    assert logits.shape == values.shape == (3, 4)
    model.train();assert model.policy.training and not model.world.training
    assert not any(p.requires_grad for p in model.world.parameters())


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
@pytest.mark.parametrize('architecture', ['vit','cnn2x2'])
def test_only_latent_dynamics_collect_and_ppo_changes_policy_not_world(monkeypatch, device, architecture):
    setup(13, 'cpu');model = tiny_model(architecture).to(device)
    from gym2048_env import Gym2048Env
    def forbidden(*args, **kwargs):
        pytest.fail('Real transition, decoder or behavior must never enter imagined rollout')
    monkeypatch.setattr(Gym2048Env, 'step', forbidden)
    monkeypatch.setattr(model.world, 'decode', forbidden)
    monkeypatch.setattr(model.world.tokenizer, 'decode', forbidden)
    initial = deepcopy(model.state_dict());fingerprint = world_fingerprint(model.world)
    rollout = collect_imagined(model, 2, .999, 123, 10, .5, 4)
    assert rollout.states.shape == (8, 4, 16, 4)
    assert all(r['steps'] == 4 and r['truncated'] and not r['terminated'] for r in rollout.episodes)
    assert all(r['spawn_return'] == 8 for r in rollout.episodes)
    assert not rollout.states.requires_grad
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    metrics = update(model.policy, optimizer, rollout, epochs=2, batch_size=4, target_kl=0)
    assert metrics['kl_method'] == 'masked_log_prob_v1'
    assert metrics['updates'] == 4 and np.isfinite(metrics['rollout_kl'])
    trainer.assert_frozen(model, fingerprint)
    for key, value in initial.items():
        if key.startswith('world.'):
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    for head in ('policy_head', 'value_head'):
        assert not torch.equal(initial[f'policy.{head}.weight'], model.state_dict()[f'policy.{head}.weight'])


def test_targets_keep_terminal_and_truncated_tails_separate():
    def row(reward, value):
        return (torch.zeros(4,16,4), torch.ones(4,dtype=torch.bool), torch.tensor(0),
                torch.tensor(0.), torch.zeros(4), torch.tensor(value), reward)
    records = [[row(1., 10.), row(2., 20.)], [row(3., 30.)]]
    result = pack_records(records, [torch.tensor(0.), torch.tensor(40.)], [], 'cpu', .9, 2, .5)
    # First episode terminates; second has an artificial time limit and V=40.
    torch.testing.assert_close(result[6], torch.tensor([10.9, 2., 39.]))
    torch.testing.assert_close(result[5], torch.tensor([.9, -18., 9.]))


def test_predicted_terminal_keeps_last_spawn_reward_and_zero_bootstrap(monkeypatch):
    setup(4, 'cpu');model = tiny_model()
    original = model.world.predict_outputs
    def terminal(u, events, nxt):
        reward, done, legal = original(u, events, nxt)
        return reward, torch.ones_like(done) * 10, legal
    monkeypatch.setattr(model.world, 'predict_outputs', terminal)
    result = collect_games(model, 2, .999, 12, max_steps=5)
    assert result[0].shape[0] == 2
    torch.testing.assert_close(result[6], torch.full((2,), 2 / 128))
    assert all(r['terminated'] and not r['truncated'] and r['spawn_return'] == 2 for r in result[7])


def test_parallel_games_match_serial_and_do_not_advance_torch_rng():
    setup(4, 'cpu');model = tiny_model()
    before = torch.get_rng_state().clone()
    serial = collect_imagined(model, 4, .99, 911, max_steps=3)
    with GamePool(2) as pool:
        parallel = collect_imagined(model, 4, .99, 911, max_steps=3, pool=pool)
    for key in ('states', 'masks', 'actions', 'old_log_probs', 'old_logits', 'advantages', 'returns'):
        torch.testing.assert_close(getattr(serial, key), getattr(parallel, key), rtol=2e-5, atol=2e-5)
    assert serial.episodes == parallel.episodes
    assert torch.equal(before, torch.get_rng_state())


def test_world_gate_rejects_failed_or_tampered_source(verified_fixture, monkeypatch):
    source, provenance = trainer.verified_source(verified_fixture)
    assert provenance['world_sha256'] == world_fingerprint(model_from_checkpoint(source).world)
    import algorithms.latent_imagination.world.source as policy
    original = policy.json.loads
    def failed(text):
        result = original(text)
        if isinstance(result, dict) and 'games' in result:
            result['passed'] = False
        return result
    monkeypatch.setattr(policy.json, 'loads', failed)
    with pytest.raises(ValueError, match='reports do not match'):
        trainer.verified_source(verified_fixture)


def fake_evaluation(*args, **kwargs):
    return dict(mean_return=100., mean_steps=45., episodes=1, seed=1, max_tile=128,
                p512=0., p1024=0., p2048=0., results=[])


def test_cli_cpu_resume_matches_and_skips_behavior(verified_fixture, tmp_path, monkeypatch):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **kw: None)
    monkeypatch.setattr(trainer, 'evaluate', fake_evaluation)
    options = ['--world-run', str(verified_fixture), '--device', 'cpu', '--workers', '1',
        '--episodes-per-update', '2', '--max-imagined-steps', '2', '--eval-every', '1', '--plot-every', '1']
    whole, resumed = tmp_path / 'whole', tmp_path / 'resumed'
    trainer.main(['--iterations', '2', '--save-dir', str(whole), *options])
    trainer.main(['--iterations', '1', '--save-dir', str(resumed), *options])
    trainer.main(['--iterations', '2', '--resume', str(resumed / 'last.pt')])
    a, b = read_checkpoint(whole / 'last.pt'), read_checkpoint(resumed / 'last.pt')
    assert a['algorithm'] == b['algorithm'] == 'latent_afterstate_ppo'
    for key, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][key], atol=0, rtol=0)
    for i, values in a['optimizer']['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value, b['optimizer']['state'][i][key], atol=0, rtol=0)
    assert a['verified_world_sha256'] == a['world_source']['world_sha256']
    assert not (resumed / 'behavior_data').exists()
    for key, value in REFERENCE_DEFAULTS.items():
        if key not in ('episodes_per_update', 'architecture'):
            assert a['config'][key] == value
    assert a['config']['architecture'] == 'vit'
    assert len(a['optimizer']['param_groups'][0]['params']) == len(list(model_from_checkpoint(a).policy.parameters()))
    trainer.assert_frozen(model_from_checkpoint(b), b['verified_world_sha256'])
    with pytest.raises(SystemExit):
        trainer.main(['--iterations', '3', '--resume', str(resumed / 'last.pt'), '--gamma', '.9'])


def test_saved_policy_real_evaluation_and_play_use_learned_candidates(tmp_path, monkeypatch):
    from common.evaluation import evaluate
    from play import make_agent
    from gym2048_env import Gym2048Env
    import gym2048_afterstate_env as exact
    setup(1, 'cpu');model = tiny_model()
    monkeypatch.setattr(exact, 'candidate_afterstates', lambda *a, **kw: pytest.fail('No exact move oracle in policy'))
    cp = tmp_path / 'model.pt'
    save_checkpoint(cp, model, torch.optim.Adam(model.policy.parameters()), 1, 'latent_afterstate_ppo', {}, 0.)
    export_model(cp, tmp_path / 'export.pt')
    controller, _, metadata = make_agent('latent_afterstate_ppo', tmp_path / 'export.pt')
    env = Gym2048Env();board, info = env.reset(seed=31)
    assert info['can_move_dir'][controller(board, info)]
    result = evaluate(model, 2, 32)
    assert result['episodes'] == 2 and result['mean_steps'] > 0 and 'p8192' in result
    assert metadata['transition_model'] == 'frozen_neural_afterstate'


def test_pipeline_entry_routes_directly_to_ppo(monkeypatch):
    from algorithms.latent_imagination import pipeline
    captured = []
    monkeypatch.setattr(trainer, 'main', lambda argv: captured.append(argv) or True)
    assert pipeline.main(['imagine', '--iterations', '3'])
    assert captured == [['--iterations', '3']]


def test_cnn2x2_serialization_and_candidate_sharing(tmp_path):
    setup(4,'cpu');model=tiny_model('cnn2x2')
    cells=torch.randn(2,4,16,4);order=[3,1,0,2]
    a,b=model.policy(cells);c,d=model.policy(cells[:,order])
    torch.testing.assert_close(c,a[:,order]);torch.testing.assert_close(d,b[:,order])
    convs=[m for m in model.policy.modules() if isinstance(m,torch.nn.Conv2d)]
    assert [(m.in_channels,m.out_channels,m.kernel_size) for m in convs]==[(4,64,(2,2)),(64,128,(2,2)),(4,128,(1,1))]
    path=tmp_path/'cnn.pt';save_checkpoint(path,model,torch.optim.Adam(model.policy.parameters()),1,'latent_afterstate_ppo',{},0.)
    restored=model_from_checkpoint(read_checkpoint(path))
    assert restored.architecture=='cnn2x2'
    for x,y in zip(model.policy(cells),restored.policy(cells)):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_lr_pressure_room_bounds_and_resume():
    from algorithms.latent_imagination.lr_control import KLRateController
    low=dict(rollout_kl=.019,update_stop_reason='target_kl',effective_epochs=.2)
    high=dict(rollout_kl=.001,update_stop_reason='epochs_complete',effective_epochs=4.)
    c=KLRateController('adaptive_kl',1e-4,minimum=5e-5,maximum=2e-4,patience=2)
    assert c.advance(low,.02)['lr_adjustment']=='hold'
    resumed=KLRateController('adaptive_kl',1e-4,5e-5,2e-4,2,c.state_dict())
    for metrics in [low]*20+[high]*40:
        assert c.advance(metrics,.02)==resumed.advance(metrics,.02)
        assert 5e-5<=c.lr<=2e-4
    assert c.lr==2e-4
    with pytest.raises(ValueError):c.advance(low,0)


def test_adaptive_lr_checkpoint_continuation_and_legacy_default(verified_fixture,tmp_path,monkeypatch):
    from algorithms.latent_imagination import lr_control
    monkeypatch.setattr(trainer,'evaluate',fake_evaluation)
    monkeypatch.setattr(lr_control,'plot_control',lambda *a:None)
    original=trainer.update
    def pressure(*args,**kwargs):
        metrics=original(*args,**kwargs)
        metrics.update(rollout_kl=.04,update_stop_reason='target_kl',effective_epochs=.2)
        return metrics
    monkeypatch.setattr(trainer,'update',pressure)
    options=['--world-run',str(verified_fixture),'--device','cpu','--workers','1',
        '--episodes-per-update','2','--max-imagined-steps','2','--lr-patience','1','--eval-every','1']
    whole,part=tmp_path/'whole',tmp_path/'part'
    trainer.main(['--iterations','3','--save-dir',str(whole),*options])
    trainer.main(['--iterations','1','--save-dir',str(part),*options])
    first=read_checkpoint(part/'last.pt');assert first['lr_controller']['lr']==pytest.approx(2.4e-4)
    trainer.main(['--iterations','3','--resume',str(part/'last.pt')])
    a,b=read_checkpoint(whole/'last.pt'),read_checkpoint(part/'last.pt')
    assert a['lr_controller']==b['lr_controller']
    assert b['optimizer']['param_groups'][0]['lr']==pytest.approx(3e-4*.8**3)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    # A legacy checkpoint does not silently acquire a new schedule on resume.
    legacy=tmp_path/'legacy.pt';b.pop('lr_controller')
    for key in ('lr_schedule','lr_min','lr_max','lr_patience'):b['config'].pop(key)
    torch.save(b,legacy)
    trainer.main(['--iterations','4','--resume',str(legacy),'--save-dir',str(tmp_path/'legacy-run')])
    assert read_checkpoint(tmp_path/'legacy-run/last.pt')['config']['lr_schedule']=='constant'
