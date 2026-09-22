"""Backend routing is tested independently of the host's installed GPUs."""
import pytest
import torch

from common.training import resolve_device, resolve_args, training_parser


@pytest.mark.parametrize('cuda,mps,expected', [
    (True, True, 'cuda'), (True, False, 'cuda'),
    (False, True, 'mps'), (False, False, 'cpu'),
])
def test_auto_device_priority_and_fallback(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    assert str(resolve_device()) == expected


@pytest.mark.parametrize('device', ['cpu', 'mps', 'cuda', 'cuda:1'])
def test_explicit_device_does_not_probe_or_override_backends(monkeypatch, device):
    def unexpected_probe():
        pytest.fail('Explicit devices must not be silently replaced')
    monkeypatch.setattr(torch.cuda, 'is_available', unexpected_probe)
    monkeypatch.setattr(torch.backends.mps, 'is_available', unexpected_probe)
    assert str(resolve_device(device)) == device


def test_new_training_auto_and_resume_device_choices(monkeypatch):
    from common import training
    args = resolve_args(training_parser('test'), 'a2c', {}, ['--iterations', '2'])
    assert args.device == 'auto'
    saved = dict(algorithm='a2c', optimizer={}, config={'device': 'cpu'}, model_config={
        'obs_dim': 16, 'num_actions': 4, 'architecture': 'cnn2x2', 'encoder_version': 3})
    monkeypatch.setattr(training, 'read_checkpoint', lambda path: saved)
    options = ['--iterations', '2', '--resume', 'old.pt']
    args = resolve_args(training_parser('test'), 'a2c', {}, options)
    assert args.device == 'cpu'
    args = resolve_args(training_parser('test'), 'a2c', {}, options + ['--device', 'auto'])
    assert args.device == 'auto'
    saved['config']['device'] = 'auto'
    args = resolve_args(training_parser('test'), 'a2c', {}, options)
    assert args.device == 'auto'
