"""Versioned warm starts, optimizer phases and plateau diagnostics."""
import json
from pathlib import Path
import shutil

import torch

from common.checkpoints import read_checkpoint,model_from_checkpoint
from ..runtime import file_digest
from .model import WorldModel
from .data import atomic_json

DEFAULTS=dict(init_from='',data_from='',head_families=12000,head_batch_size=64,
    dynamics_lr=3e-5,repair_updates=2000,plateau_patience=10000,stop_on_plateau=0,train_probe_samples=128)


def initializer(root,cfg):
    if not cfg['init_from']: return None
    directory=root/'initialization';directory.mkdir(exist_ok=True)
    local=directory/'source.pt';manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        report=json.loads(manifest_path.read_text())
        if report['sha256']!=file_digest(local): raise ValueError('Initialization snapshot changed')
        return local
    source=Path(cfg['init_from']).resolve()
    if root==source.parent or root in source.parents: raise ValueError('Warm start requires a separate run directory')
    temporary=directory/'source.tmp';shutil.copyfile(source,temporary)
    checkpoint=read_checkpoint(temporary)
    if checkpoint['model_config'].get('model_version')!=2: raise ValueError('Warm start requires a neural world v2 checkpoint')
    model_from_checkpoint(checkpoint)  # Validate architecture before committing.
    temporary.replace(local)
    atomic_json(manifest_path,dict(source=str(source),iteration=checkpoint['iteration'],
        sha256=file_digest(local),note='Weights only warm start; optimizer, counters and old gates are not imported.'))
    return local


def load_initial_model(root,cfg,device):
    path=initializer(root,cfg)
    if path is None:
        return WorldModel(model_version=2,width=cfg['width'],latent_dim=cfg['latent_dim'],
            tokenizer_layers=cfg['layers'],dynamics_layers=cfg['layers'],heads=cfg['heads'],state_head_version=2).to(device)
    source=read_checkpoint(path)
    config={k:v for k,v in source['model_config'].items() if k!='model_type'}
    expected=dict(width=cfg['width'],latent_dim=cfg['latent_dim'],tokenizer_layers=cfg['layers'],
                  dynamics_layers=cfg['layers'],heads=cfg['heads'])
    if any(config[k]!=v for k,v in expected.items()):
        raise ValueError('Warm-start dimensions differ; pass the source width/latent-dim/layers/heads')
    config['state_head_version']=2;model=WorldModel(**config)
    result=model.load_state_dict(source['model'],strict=False)
    if result.unexpected_keys or any(not key.startswith('world.state_refiner.') for key in result.missing_keys):
        raise ValueError('Unsupported warm-start weight migration')
    return model.to(device)


def copy_base_data(root,cfg):
    if not cfg['data_from'] or (root/'data').exists(): return
    source=Path(cfg['data_from']).resolve()
    temporary=root/'.data-import'
    if temporary.exists(): shutil.rmtree(temporary)
    shutil.copytree(source,temporary);temporary.replace(root/'data')


def optimizer_for(model,cfg,stage):
    if stage=='tokenizer':
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'])
    # Keep group membership stable across the frozen repair prefix and resumes.
    prefixes=('action_transition.','event_transition.','action_embedding.','event_embedding.')
    dynamics=[];heads=[]
    for name,param in model.world.named_parameters():
        if not param.requires_grad: continue
        (dynamics if name.startswith(prefixes) else heads).append(param)
    return torch.optim.Adam([dict(params=heads,lr=cfg['lr'],role='heads'),
                             dict(params=dynamics,lr=cfg['dynamics_lr'],role='dynamics')])


def set_repair_phase(optimizer,stage,iteration,cfg):
    frozen=stage=='world1' and bool(cfg['init_from']) and iteration<cfg['repair_updates']
    for group in optimizer.param_groups:
        if group.get('role')=='dynamics':
            for param in group['params']: param.requires_grad_(not frozen)
    return frozen


def gate_margin(report):
    # Positive iff each requirement passes. Worst normalized deficit identifies
    # the actual bottleneck, not the average of mostly solved metrics.
    values=[]
    for row in report['checks'].values():
        value=row['value'];threshold=row['threshold']
        if value is None: return -1e9
        values.append((value-threshold if row['direction']=='min' else threshold-value)/max(threshold,.01))
    return min(values)


def plateau_state(history,iteration,patience):
    best=-float('inf');last_improvement=0
    for row in history:
        if 'gate_margin' in row and row['gate_margin']>best+1e-3:
            best=row['gate_margin'];last_improvement=row['iteration']
    return dict(stalled=bool(patience and iteration-last_improvement>=patience),
                last_improvement=last_improvement,updates_without_improvement=iteration-last_improvement)
