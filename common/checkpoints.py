"""Training checkpoint I/O and small, weights-only demo exports."""
from pathlib import Path
import random
import numpy as np
import torch
from common.models import model_from_config, validate_encoder_version, REWARD_OBJECTIVE

FORMAT_VERSION = 3
SUPPORTED_REWARD_OBJECTIVES = {REWARD_OBJECTIVE}


def save_checkpoint(path, model, optimizer, iteration, algorithm, config, best_metric, extra=None,
                    *, reward_objective=REWARD_OBJECTIVE):
    if reward_objective not in SUPPORTED_REWARD_OBJECTIVES:
        raise ValueError(f'Unsupported reward objective: {reward_objective}')
    data = dict(format_version=FORMAT_VERSION, reward_objective=reward_objective, model=model.state_dict(),
                model_config=model.model_config,
                optimizer=optimizer.state_dict(), iteration=iteration, algorithm=algorithm,
                config=config, best_metric=best_metric, rng_py=random.getstate(),
                rng_np=np.random.get_state(), rng_torch=torch.get_rng_state())
    if torch.cuda.is_available():
        data['rng_cuda'] = torch.cuda.get_rng_state_all()
    if next(model.parameters()).device.type == 'mps':
        data['rng_mps'] = torch.mps.get_rng_state()
    if extra:
        data.update(extra)
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def read_checkpoint(path):
    # Only load locally generated/trusted training checkpoints (contains RNG data).
    data = torch.load(path, map_location='cpu', weights_only=False)
    if (data.get('format_version') != FORMAT_VERSION
            or data.get('reward_objective') not in SUPPORTED_REWARD_OBJECTIVES):
        raise ValueError('Checkpoint reward objective/format is incompatible')
    return data


def checkpoint_model_config(data):
    config = data['model_config']
    validate_encoder_version(config)
    return config


def restore_checkpoint(data, model, optimizer=None, restore_rng=False):
    if model.model_config != checkpoint_model_config(data):
        raise ValueError('Cannot resume into a different architecture; start a separate model experiment')
    model.load_state_dict(data['model'])
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer'])
    if restore_rng:
        random.setstate(data['rng_py']); np.random.set_state(data['rng_np'])
        torch.set_rng_state(data['rng_torch'])
        if 'rng_cuda' in data and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(data['rng_cuda'])
        if 'rng_mps' in data and next(model.parameters()).device.type == 'mps':
            torch.mps.set_rng_state(data['rng_mps'])
    return data


def load_checkpoint(path, model, optimizer=None, restore_rng=False):
    return restore_checkpoint(read_checkpoint(path), model, optimizer, restore_rng)


def model_from_checkpoint(data, device='cpu'):
    model = model_from_config(checkpoint_model_config(data)).to(device)
    restore_checkpoint(data, model)
    return model


def export_model(source, output):
    """Keep weights and provenance; omit optimizer, RNG and replay buffer."""
    data = read_checkpoint(source)
    exported = {key: data[key] for key in ('format_version', 'reward_objective', 'algorithm',
                                          'iteration', 'model', 'best_metric')}
    exported['model_config'] = checkpoint_model_config(data)
    exported['config'] = {key: data['config'][key] for key in
                         ('gamma', 'eval_seed', 'eval_episodes', 'eval_mcts_sims', 'search_depth')
                         if key in data['config']}
    exported['inference_only'] = True
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    return exported


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_model(args.input, args.output)
    print(f'Exported {args.output}')
