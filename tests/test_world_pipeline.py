"""World construction must finish every stage before policy use."""
import json
from pathlib import Path

import pytest
import torch

from algorithms.latent_imagination.world import pipeline
from algorithms.latent_imagination.world.source import verified_source
from algorithms.latent_imagination.world.data import atomic_json
from test_latent_imagination import verified_fixture


def fixture_pipeline(tmp_path, monkeypatch, *, external=False):
    root = tmp_path / 'build'
    root.mkdir()
    policy = tmp_path / 'policy.pt'
    torch.save(dict(format_version=3, reward_objective='spawn_mass', iteration=10000, algorithm='latent_afterstate_ppo', verified_world_sha256='world'), policy)
    cfg = {**pipeline.DEFAULTS, 'device': 'cpu', 'workers': 1, 'policy_checkpoint': str(policy)}
    if external:
        cfg['base_world'] = str(tmp_path / 'external')
    state = dict(version=1, config=cfg, status='running', stages={})
    calls = []
    monkeypatch.setattr(pipeline, 'verified_source', lambda path: (
        {}, dict(world_run=str(path), world_sha256='world')))
    monkeypatch.setattr(pipeline.base, 'main', lambda argv: calls.append(('base', argv)) or True)
    monkeypatch.setattr(pipeline.trajectories, 'main', lambda argv: calls.append(('trajectories', argv)) or True)
    return root, cfg, state, calls


def test_world_runs_base_then_trajectory_training_and_records_completion(tmp_path, monkeypatch):
    root, cfg, state, calls = fixture_pipeline(tmp_path, monkeypatch)
    assert pipeline.build(root, cfg, state)
    assert [name for name, _ in calls] == ['base', 'trajectories']
    assert all(state['stages'][stage]['status'] == 'passed' for stage in pipeline.STAGES)
    assert state['status'] == 'complete'
    args = calls[-1][1]
    assert args[args.index('--world-run') + 1] == str(root / 'base')
    assert args[args.index('--run-dir') + 1] == str(root / 'world')


def test_exploration_is_trained_and_resumed_before_trajectory_fitting(tmp_path, monkeypatch):
    from algorithms.latent_imagination import train
    root, cfg, state, calls = fixture_pipeline(tmp_path, monkeypatch, external=True)
    cfg['policy_checkpoint'] = ''
    policy = root / 'exploration/last.pt'
    policy.parent.mkdir()
    torch.save(dict(format_version=3, reward_objective='spawn_mass', iteration=10), policy)
    def bootstrap(argv):
        calls.append(('exploration', argv))
        torch.save(dict(format_version=3, reward_objective='spawn_mass', iteration=cfg['bootstrap_iterations'], algorithm='latent_afterstate_ppo',
                        verified_world_sha256='world'), policy)
    monkeypatch.setattr(train, 'main', bootstrap)
    assert pipeline.build(root, cfg, state)
    assert [name for name, _ in calls] == ['exploration', 'trajectories']
    assert '--resume' in calls[0][1]
    calls.clear()
    assert pipeline.build(root, cfg, state)
    assert [name for name, _ in calls] == ['trajectories']


def test_failed_base_and_unrelated_policy_stop_later_stages(tmp_path, monkeypatch):
    root, cfg, state, calls = fixture_pipeline(tmp_path, monkeypatch)
    def paused(argv):
        raise SystemExit(2)
    monkeypatch.setattr(pipeline.base, 'main', paused)
    assert not pipeline.build(root, cfg, state)
    assert calls == [] and state['status'] == 'paused'
    cfg['base_world'] = str(tmp_path / 'external')
    torch.save(dict(format_version=3, reward_objective='spawn_mass', algorithm='latent_afterstate_ppo', verified_world_sha256='different'), cfg['policy_checkpoint'])
    with pytest.raises(ValueError, match='this base world'):
        pipeline.build(root, cfg, state)
    assert calls == []


def test_failed_trajectory_stage_leaves_world_unpublished(tmp_path, monkeypatch):
    root, cfg, state, _ = fixture_pipeline(tmp_path, monkeypatch, external=True)
    monkeypatch.setattr(pipeline.trajectories, 'main', lambda argv: False)
    assert not pipeline.build(root, cfg, state)
    assert state['status'] == 'paused' and state['stages']['trajectories']['status'] == 'paused'
    assert 'world_sha256' not in state


def test_world_source_requires_completed_pipeline_and_matching_identity(verified_fixture, tmp_path):
    root = tmp_path / 'construction'
    root.mkdir()
    (root / 'world').symlink_to(verified_fixture, target_is_directory=True)
    _, actual = verified_source(verified_fixture)
    state = dict(version=1, status='running', stages={}, world_sha256=actual['world_sha256'])
    atomic_json(root / 'world_pipeline.json', state)
    with pytest.raises(ValueError, match='incomplete'):
        verified_source(root)
    state.update(status='complete', stages={stage: dict(status='passed') for stage in pipeline.STAGES})
    atomic_json(root / 'world_pipeline.json', state)
    _, accepted = verified_source(root)
    assert accepted['world_sha256'] == actual['world_sha256']
    state['world_sha256'] = 'different'
    atomic_json(root / 'world_pipeline.json', state)
    with pytest.raises(ValueError, match='identity'):
        verified_source(root)
