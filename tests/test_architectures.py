"""Architecture experiments preserve policy masking, learning, and checkpointing."""
import numpy as np
import pytest
import torch
from algorithms import a2c
from common.models import (ActorCritic, ARCHITECTURES, ValidResidualConvBlock,
                           RotaryPosition2D, masked_categorical, model_from_config)
from common.training import setup
from common.rollout import collect_actor_critic
from common.parallel import GamePool, model_snapshot, worker_model
from common.checkpoints import (save_checkpoint, read_checkpoint, model_from_checkpoint,
                                export_model, load_checkpoint)
from gym2048_env import Gym2048Env


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_new_models_batch_modes_masks_and_gradients(architecture):
    setup(31)
    model = ActorCritic(architecture=architecture)
    states = torch.stack((torch.arange(16).float(), torch.zeros(16), torch.full((16,), 31.)))
    logits, values = model(states)
    assert logits.shape == (3, 4) and values.shape == (3,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    one_logits, one_value = model(states[0])
    torch.testing.assert_close(one_logits, logits[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(one_value, values[0], atol=1e-6, rtol=1e-5)
    model.eval()
    with torch.no_grad():
        eval_logits, eval_values = model(states)
    torch.testing.assert_close(eval_logits, logits, atol=0, rtol=0)
    torch.testing.assert_close(eval_values, values, atol=0, rtol=0)
    dist = masked_categorical(logits, [True, False, True, False])
    assert (dist.probs[:, [1, 3]] == 0).all()
    loss = -dist.log_prob(torch.zeros(3, dtype=torch.long)).mean() + (values - 1).square().mean()
    loss.backward()
    assert model.policy_head.weight.grad.abs().sum() > 0
    assert model.value_head.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0
    if architecture == 'vit':
        assert model.trunk.blocks[0].qkv.weight.grad.abs().sum() > 0
        assert model.trunk.cell_projection.weight.grad.abs().sum() > 0
    else:
        assert model.trunk[0].layers[0].weight.grad.abs().sum() > 0
        assert model.trunk[0].shortcut[1].weight.grad.abs().sum() > 0


def test_vit_uses_positions_and_all_cells_including_empty():
    setup(32)
    model = ActorCritic(architecture='vit')
    x = torch.tensor([1., 2., 0., 3., 0., 4., 5., 0., 6., 7., 0., 8., 9., 0., 10., 0.], requires_grad=True)
    value = model(x)[1]
    value.backward()
    # The numeric rank channel gives a gradient path to all 16 positions.
    assert (x.grad.abs() > 0).all()
    assert not torch.allclose(value, model(x.detach().flip(0))[1], atol=1e-6, rtol=1e-6)


def test_vit_rope_preserves_norm_and_encodes_signed_2d_relative_positions():
    setup(51)
    rotary = RotaryPosition2D()
    # Identical content at every cell isolates positional effects on attention.
    q = torch.randn(1, 4, 1, 24).expand(-1, -1, 16, -1)
    k = torch.randn(1, 4, 1, 24).expand_as(q)
    rotated_q, rotated_k = rotary(q), rotary(k)
    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(rotated_q[:, :, 0], q[:, :, 0])  # Origin has zero rotation.
    torch.testing.assert_close(rotated_q[:, :, 0, :12], rotated_q[:, :, 1, :12])  # Same row.
    torch.testing.assert_close(rotated_q[:, :, 0, 12:], rotated_q[:, :, 4, 12:])  # Same column.
    scores = rotated_q @ rotated_k.transpose(-1, -2)
    torch.testing.assert_close(scores[..., 0, 1], scores[..., 5, 6])
    torch.testing.assert_close(scores[..., 0, 1], scores[..., 14, 15])
    assert not torch.allclose(scores[..., 5, 6], scores[..., 5, 4])  # Right versus left.
    assert not torch.allclose(scores[..., 5, 1], scores[..., 5, 9])  # Up versus down.
    assert not torch.allclose(scores[..., 0, 1], scores[..., 3, 4])  # No row wrap.
    assert not list(rotary.parameters())  # Position rotations are fixed, not learned.
    with pytest.raises(ValueError, match='divisible by four'):
        RotaryPosition2D(head_dim=22)


def test_vit_ordered_readout_preserves_positions_without_attention():
    setup(51)
    model = ActorCritic(architecture='vit')
    # Isolate the residual identity paths: the readout must still distinguish
    # a permutation of the same tiles, without help from learned attention.
    with torch.no_grad():
        for block in model.trunk.blocks:
            block.attention_output.weight.zero_()
            block.attention_output.bias.zero_()
            block.mlp[-1].weight.zero_()
            block.mlp[-1].bias.zero_()
    state = torch.arange(16).float()
    assert not torch.allclose(model(state)[1], model(state.flip(0))[1], atol=1e-6, rtol=1e-6)
    assert not hasattr(model.trunk, 'cls_token')
    assert not hasattr(model.trunk, 'position_embedding')


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_new_architecture_checkpoint_export_and_worker_snapshot(tmp_path, architecture):
    from play import make_agent
    setup(4)
    model = ActorCritic(architecture=architecture)
    snapshot = worker_model(model_snapshot(model))
    states = torch.arange(32).float().reshape(2, 16)
    torch.testing.assert_close(model(states)[0], snapshot(states)[0], atol=0, rtol=0)
    full, exported = tmp_path / 'last.pt', tmp_path / 'model.pt'
    save_checkpoint(full, model, torch.optim.Adam(model.parameters()), 1, 'a2c', {}, 0.)
    restored = model_from_checkpoint(read_checkpoint(full))
    torch.testing.assert_close(model(states)[1], restored(states)[1], atol=0, rtol=0)
    with pytest.raises(ValueError, match='different architecture'):
        load_checkpoint(full, ActorCritic(architecture='vit' if architecture == 'cnn2x2' else 'cnn2x2'))
    export_model(full, exported)
    exported_model = model_from_checkpoint(torch.load(exported, weights_only=True))
    torch.testing.assert_close(model(states)[0], exported_model(states)[0], atol=0, rtol=0)
    agent, label, metadata = make_agent('a2c', exported)
    env = Gym2048Env()
    state, info = env.reset(seed=9)
    assert info['can_move_dir'][agent(state, info)]
    assert architecture.upper() in label
    assert metadata['model_config']['architecture'] == architecture


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_new_models_real_a2c_update_and_unchanged_policy_ratio(architecture):
    setup(11)
    model = ActorCritic(architecture=architecture)
    rollout = collect_actor_critic(Gym2048Env(), model, 2, 1., 813)
    dist = masked_categorical(model(rollout.states)[0], rollout.masks)
    ratios = (dist.log_prob(rollout.actions) - rollout.old_log_probs).exp()
    torch.testing.assert_close(ratios, torch.ones_like(ratios), atol=1e-6, rtol=1e-6)
    before = model.policy_head.weight.detach().clone()
    result = a2c.update(model, torch.optim.Adam(model.parameters(), lr=3e-4), rollout)
    assert result['updates'] == 1
    assert not torch.equal(before, model.policy_head.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_new_models_parallel_a2c_resume(tmp_path, monkeypatch, architecture):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--device', 'cpu', '--architecture', architecture, '--episodes-per-update', '2', '--workers', '2',
               '--eval-episodes', '2', '--eval-every', '1', '--seed', '7']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    a2c.main(['--iterations', '2', '--save-dir', str(whole), *options])
    a2c.main(['--iterations', '1', '--save-dir', str(split), *options])
    a2c.main(['--iterations', '2', '--resume', str(split / 'last.pt'), '--workers', '2'])
    expected, actual = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert actual['model_config']['architecture'] == architecture
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)


@pytest.mark.parametrize('architecture,count', [('cnn2x2', 173893), ('vit', 187085)])
def test_documented_architecture_parameter_counts(architecture, count):
    assert architecture in ARCHITECTURES
    assert sum(p.numel() for p in ActorCritic(architecture=architecture).parameters()) == count


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_new_models_reject_non_4x4_board(architecture):
    with pytest.raises(ValueError, match='4x4'):
        ActorCritic(obs_dim=9, architecture=architecture)


def test_cnn2x2_valid_convolutions_normalization_and_activation(monkeypatch):
    setup(41)
    small = ActorCritic(architecture='cnn2x2')
    block = small.trunk[0]
    assert isinstance(block, ValidResidualConvBlock)
    assert [type(layer) for layer in block.layers] == [torch.nn.Conv2d, torch.nn.GroupNorm, torch.nn.SiLU, torch.nn.Conv2d, torch.nn.GroupNorm]
    main_convs = [layer for layer in block.layers if isinstance(layer, torch.nn.Conv2d)]
    assert [(layer.in_channels, layer.out_channels) for layer in main_convs] == [(17, 64), (64, 128)]
    assert all(layer.kernel_size == (2, 2) and layer.stride == (1, 1) for layer in main_convs)
    for layer in small.modules():
        if isinstance(layer, torch.nn.Conv2d):
            assert layer.padding == (0, 0)
        if isinstance(layer, torch.nn.GroupNorm):
            assert layer.num_groups == 4
            assert layer.eps == 1e-5 and layer.affine
    assert small.trunk[2].in_features == 512 and small.trunk[2].out_features == 256
    assert isinstance(small.trunk[3], torch.nn.LayerNorm)
    assert isinstance(small.trunk[4], torch.nn.SiLU)
    def forbidden(*args, **kwargs):
        raise AssertionError('CNN2x2 must not pad the main path or the shortcut')
    monkeypatch.setattr(torch.nn.functional, 'pad', forbidden)
    shapes, hooks = [], []
    for layer in main_convs:
        hooks.append(layer.register_forward_hook(lambda _m, _i, output: shapes.append(output.shape)))
    assert small.features(torch.zeros(2, 16)).shape == (2, 256)
    for hook in hooks:
        hook.remove()
    assert shapes == [torch.Size([2, 64, 3, 3]), torch.Size([2, 128, 2, 2])]
    assert block.shortcut(torch.zeros(2, 17, 4, 4)).shape == (2, 128, 2, 2)


def test_shrinking_residual_shortcut_covers_the_board_and_carries_gradients():
    block = ValidResidualConvBlock()
    board = torch.arange(16).float().reshape(1, 1, 4, 4)
    # A 3x3, stride-one average maps 4x4 -> 2x2 without discarding a border.
    torch.testing.assert_close(block.shortcut[0](board)[0, 0], torch.tensor([[5., 6.], [9., 10.]]))
    with torch.no_grad():
        for layer in block.layers:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.zero_()
                layer.bias.zero_()
    x = torch.randn(2, 17, 4, 4, requires_grad=True)
    output = block(x)
    torch.testing.assert_close(output, torch.nn.functional.silu(block.shortcut(x)), atol=0, rtol=0)
    output.sum().backward()
    assert (x.grad.abs() > 0).all()
    assert block.shortcut[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('model_type', ['actor_critic', 'muzero'])
@pytest.mark.parametrize('old_version', [None, 2])
def test_cnn2x2_version_metadata_prevents_loading_former_architecture(tmp_path, model_type, old_version):
    from algorithms.muzero import MuZeroNetwork
    model = (ActorCritic(architecture='cnn2x2') if model_type == 'actor_critic'
             else MuZeroNetwork(architecture='cnn2x2'))
    assert model.model_config['encoder_version'] == 3
    restored = model_from_config(model.model_config)
    encoder = restored if model_type == 'actor_critic' else restored.encoder
    assert encoder.trunk[0].layers[0].weight.shape == (64, 17, 2, 2)
    path = tmp_path / 'old.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1,
                    'a2c' if model_type == 'actor_critic' else 'muzero', {}, 0.)
    data = read_checkpoint(path)
    if old_version is None:
        del data['model_config']['encoder_version']
    else:
        data['model_config']['encoder_version'] = old_version
    torch.save(data, path)
    with pytest.raises(ValueError, match='new training run'):
        model_from_checkpoint(read_checkpoint(path))
    with pytest.raises(ValueError, match='new training run'):
        load_checkpoint(path, model)
    with pytest.raises(ValueError, match='new training run'):
        model_from_config(data['model_config'])


@pytest.mark.parametrize('model_type', ['actor_critic', 'muzero'])
@pytest.mark.parametrize('old_version', [None, 1])
def test_vit_version_rejects_former_cls_checkpoints(tmp_path, model_type, old_version):
    from algorithms.muzero import MuZeroNetwork
    model = (ActorCritic(architecture='vit') if model_type == 'actor_critic'
             else MuZeroNetwork(architecture='vit'))
    assert model.model_config['encoder_version'] == 2
    path = tmp_path / 'vit.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1,
                    'a2c' if model_type == 'actor_critic' else 'muzero', {}, 0.)
    data = read_checkpoint(path)
    if old_version is None:
        del data['model_config']['encoder_version']
    else:
        data['model_config']['encoder_version'] = old_version
    torch.save(data, path)
    with pytest.raises(ValueError, match='new training run'):
        model_from_checkpoint(read_checkpoint(path))
    with pytest.raises(ValueError, match='new training run'):
        load_checkpoint(path, model)
    with pytest.raises(ValueError, match='new training run'):
        model_from_config(data['model_config'])
