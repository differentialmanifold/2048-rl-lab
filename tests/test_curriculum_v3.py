"""Meaningful coverage, gradients, warm-start and optimizer-resume contracts."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.latent_imagination.world import base as pipeline
from algorithms.latent_imagination.world.boundaries import (canonical_keys,prepare_boundaries,CurriculumData,boundary_family)
from algorithms.latent_imagination.world.runtime import load_initial_model,optimizer_for,set_repair_phase,plateau_state
from algorithms.latent_imagination.world.data import generate,SyntheticData,canonical
from algorithms.latent_imagination.world.branches import full_branch_loss_chunks,evaluate,gates
from algorithms.latent_imagination.world.objectives import supervised_world_loss
from algorithms.latent_imagination.world.model import WorldModel
from algorithms.latent_imagination.world.verification import reference
from common.checkpoints import read_checkpoint,save_checkpoint,restore_checkpoint,model_from_checkpoint
from common.training import setup


@pytest.fixture(scope='module')
def curriculum(tmp_path_factory):
    setup(0,'cpu');root=tmp_path_factory.mktemp('curriculum-v3')
    generate(root/'data',families=180,horizon=10,seed=17,workers=1,max_rank=6)
    base=SyntheticData(root/'data')
    owners=prepare_boundaries(root/'head_data',base,180,17,6,workers=1)
    return root,CurriculumData(base,root/'head_data',owners)


def tiny(version=2):
    return WorldModel(model_version=2,width=16,latent_dim=4,tokenizer_layers=1,
                               dynamics_layers=1,heads=2,state_head_version=version)


def test_vectorized_canonical_matches_exact_d4_bytes():
    a=np.random.default_rng(9).integers(0,13,(400,16),dtype=np.uint8);a[:30,8:]=0
    assert canonical_keys(a)==[canonical(row) for row in a]


def test_boundary_labels_and_isolation_and_resume(curriculum):
    root,data=curriculum
    before=(root/'head_data/manifest.json').read_bytes()
    prepare_boundaries(root/'head_data',data.base,180,17,6,workers=1)
    assert before==(root/'head_data/manifest.json').read_bytes()
    owners={}
    for split,part in data.heads.items():
        assert part['dones'].any() and (~part['dones']).any()
        for b,mask,done in zip(part['boards'],part['masks'],part['dones']):
            truth=reference(b.numpy());assert truth['done']==bool(done)
            assert truth['mask']==mask.tolist()
            key=canonical(b.numpy());assert owners.get(key,split)==split;owners[key]=split
    batch=data.sample_heads('train',1000,'cpu',np.random.default_rng(89))
    for b,mask,done in zip(batch['states'],batch['masks'],batch['dones']):
        truth=reference(b.numpy());assert truth['mask']==mask.tolist() and truth['done']==bool(done)
        assert canonical(b.numpy()) not in data.forbidden


def test_windows_cover_later_steps_and_keep_real_padding(curriculum):
    _,data=curriculum;rng=np.random.default_rng(10)
    for steps in (1,3,10):
        batch=data.sample_sequence('train',512,steps,'cpu',rng);valid=batch['valid'].bool()
        assert valid[:,0].all()
        assert not (valid[:,1:] & ~valid[:,:-1]).any()
        assert not (valid[:,1:] & batch['dones'][:,:-1]).any()
        assert torch.equal(batch['states'][:,1:][valid[:,1:]],batch['next_states'][:,:-1][valid[:,1:]])
        if steps>1: assert valid[:,-1].sum()>=256
        roots={b.numpy().tobytes() for b in data.splits['train']['boards']}
        assert any(b.numpy().tobytes() not in roots for b in batch['states'][:,0])
    prior=data.sample_prior('train',128,'cpu',rng)
    assert int((prior['events']==32).sum())==32
    assert torch.allclose(prior['chance_probs'].sum(-1),torch.ones(128))
    small=[int(data.sample_prior('train',1,'cpu',rng)['events'][0]) for _ in range(64)]
    assert 32 in small and any(c!=32 for c in small)


def test_full_branch_prior_and_encoded_readout_gradients(curriculum):
    _,data=curriculum;model=tiny();model.set_phase('world');rng=np.random.default_rng(8)
    batch=data.sample_prior('train',4,'cpu',rng)
    metrics={};losses=full_branch_loss_chunks(model,batch,7,enhanced=True,metrics=metrics)
    for loss in losses: loss.backward()
    assert model.world.prior[0].weight.grad.abs().sum()>0
    assert model.world.state_refiner.output.weight.grad.abs().sum()>0
    assert 'branch_prior_ce' in metrics and 'branch_encoded_head_nll' in metrics
    # Chunk size changes memory use, not the objective/normalization.
    a=deepcopy(model);b=deepcopy(model);a.zero_grad();b.zero_grad()
    for loss in full_branch_loss_chunks(a,batch,7,enhanced=True): loss.backward()
    for loss in full_branch_loss_chunks(b,batch,10000,enhanced=True): loss.backward()
    for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
        if pa.grad is not None: torch.testing.assert_close(pa.grad,pb.grad,rtol=5e-4,atol=3e-4,msg=name)


def test_migration_preserves_existing_predictions_and_freeze_then_thaw(tmp_path):
    setup(0,'cpu');old=tiny(1);checkpoint=tmp_path/'source.pt'
    save_checkpoint(checkpoint,old,torch.optim.Adam(old.parameters()),360000,'latent_dreamer',{},0)
    newdir=tmp_path/'new';newdir.mkdir()
    cfg={**pipeline.DEFAULTS,'init_from':str(checkpoint),'width':16,'latent_dim':4,'layers':1,'heads':2,'repair_updates':1}
    model=load_initial_model(newdir,cfg,'cpu');boards=torch.randint(0,8,(4,16));actions=torch.arange(4)
    old_z=old.world.encode(boards);new_z=model.world.encode(boards)
    torch.testing.assert_close(old_z,new_z,atol=0,rtol=0)
    for x,y in zip(old.world.state_heads(old_z),model.world.state_heads(new_z)): torch.testing.assert_close(x,y,atol=0,rtol=0)
    torch.testing.assert_close(old.world.afterstate(old_z,actions),model.world.afterstate(new_z,actions),atol=0,rtol=0)
    model.set_phase('world');opt=optimizer_for(model,cfg,'world1')
    assert set_repair_phase(opt,'world1',0,cfg)
    assert not model.world.action_transition.output.weight.requires_grad
    assert model.world.state_refiner.output.weight.requires_grad
    assert not set_repair_phase(opt,'world1',1,cfg)
    assert model.world.action_transition.output.weight.requires_grad


def test_enhanced_supervision_and_resume_across_repair_boundary(curriculum,tmp_path):
    _,data=curriculum;cfg={**pipeline.DEFAULTS,'init_from':'fixture','repair_updates':1}
    setup(1,'cpu');initial=tiny()
    models=[deepcopy(initial),deepcopy(initial)];opts=[]
    for model in models: model.set_phase('world');opts.append(optimizer_for(model,cfg,'world1'))
    rng=np.random.default_rng(10);batches=[data.sample_sequence('train',4,3,'cpu',rng) for _ in range(2)]
    head=data.sample_heads('train',4,'cpu',rng)
    for index in range(2):
        for iteration,batch in enumerate(batches):
            model=models[index];opt=opts[index]
            set_repair_phase(opt,'world1',iteration,cfg);opt.zero_grad(set_to_none=True)
            loss,metrics=supervised_world_loss(model,batch,head,enhanced=True)
            assert torch.isfinite(loss) and 'encoded_head_nll' in metrics
            loss.backward();opt.step()
            if index==1 and iteration==0:
                cp=tmp_path/'resume.pt';save_checkpoint(cp,model,opt,1,'latent_dreamer',{},0)
                saved=read_checkpoint(cp);model=model_from_checkpoint(saved);model.set_phase('world')
                opt=optimizer_for(model,cfg,'world1');restore_checkpoint(saved,model,opt,restore_rng=True)
                models[index]=model;opts[index]=opt
    for key,value in models[0].state_dict().items(): torch.testing.assert_close(value,models[1].state_dict()[key],atol=0,rtol=0)
    for key,value in initial.world.tokenizer.state_dict().items():
        torch.testing.assert_close(value,models[0].world.tokenizer.state_dict()[key],atol=0,rtol=0)


def test_extra_gates_and_plateau_are_fail_closed():
    checks=gates('world1',{},enhanced=True)['checks']
    good={k:1. if v['direction']=='min' else 0. for k,v in checks.items()}
    assert gates('world1',good,True)['passed']
    del good['head_agreement'];assert not gates('world1',good,True)['passed']
    assert plateau_state([dict(iteration=200,gate_margin=-.5),dict(iteration=400,gate_margin=-.6)],1400,1000)['stalled']
    assert not plateau_state([dict(iteration=200,gate_margin=-.5),dict(iteration=400,gate_margin=-.2)],1000,1000)['stalled']


def test_v3_fork_preserves_both_corpora(curriculum,tmp_path,monkeypatch):
    original,data=curriculum;model=tiny();source=original/'source_v3.pt'
    save_checkpoint(source,model,torch.optim.Adam(model.parameters()),100,'latent_dreamer',{},0,
        dict(objective_version=3,dataset_sha256=data.digest,base_dataset_sha256=data.base_digest))
    cfg={**pipeline.DEFAULTS,'init_from':str(source),'data_from':str(original/'data'),
         'head_families':180,'families':180,'seed':17,'max_rank':6,'workers':1}
    monkeypatch.setattr(pipeline,'progress_plot',lambda *a,**kw:None)
    result=pipeline.prepare(tmp_path,cfg)
    assert result.digest==data.digest
    assert (tmp_path/'head_data/manifest.json').read_bytes()==(original/'head_data/manifest.json').read_bytes()


@pytest.mark.parametrize('stage', ['world1', 'world3', 'world10'])
@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_enhanced_mps_gradients_and_full_head_validation(curriculum,stage):
    _,data=curriculum;model=tiny().to('mps');model.set_phase('world');rng=np.random.default_rng(0)
    loss,metrics=supervised_world_loss(model,data.sample_sequence('train',4,3,'mps',rng),
        data.sample_heads('train',4,'mps',rng),enhanced=True)
    loss.backward();assert torch.isfinite(model.world.state_refiner.output.weight.grad).all()
    result=evaluate(model,data,stage,16,4)
    assert all(k in result for k in ('head_agreement','root_head_agreement','generated_next_legal','noop_invalid_mass'))


@pytest.mark.parametrize('stage,horizon', [('world3',3), ('world10',10)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('mps',marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(),reason='MPS unavailable'))])
def test_enhanced_rollout_prefix_remembers_earlier_errors(curriculum,monkeypatch,stage,horizon,device):
    """Head metric names must not overwrite the cumulative rollout mask.

    Inject an afterstate error for row 0 at step 1, and a next-state error
    for row 1 at step 2. Later correct boards must not reset either failure.
    """
    _,data=curriculum;model=tiny().to(device);model.train()
    sample_sequence=data.sample_sequence;decode=model.world.decode;pending=[]

    def fixed_sequence(split,count,steps,target,rng):
        batch=sample_sequence(split,max(64,count*2),steps,target,rng)
        ids=batch['valid'].bool().all(-1).nonzero().flatten()[:count]
        assert len(ids)==count
        batch={key:value[ids] for key,value in batch.items()}
        for t in range(steps):
            after=batch['afterstates'][:,t].clone();nxt=batch['next_states'][:,t].clone()
            if t==0: after[0,0]=(after[0,0]+1)%32
            if t==1: nxt[1,0]=(nxt[1,0]+1)%32
            pending.extend([after,nxt])
        return batch

    def controlled_decode(latent):
        if pending:
            return torch.nn.functional.one_hot(pending.pop(0).long(),32).float()
        return decode(latent)

    monkeypatch.setattr(data,'sample_sequence',fixed_sequence)
    monkeypatch.setattr(model.world,'decode',controlled_decode)
    result=evaluate(model,data,stage,samples=4,batch_size=4)
    assert model.training and not pending
    assert result['depth_samples']=={str(t):4 for t in range(1,horizon+1)}
    assert result['step_1_prefix']==.75
    for t in range(2,horizon+1): assert result[f'step_{t}_prefix']==.5
    assert result[f'step_{horizon}_afterstate']==1.
    assert result[f'step_{horizon}_next_state']==1.
    assert all(key in result for key in ('head_agreement','encoded_next_terminal_recall',
        'generated_next_terminal_recall','noop_invalid_mass','changed_invalid_mass'))
