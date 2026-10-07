"""Probability calibration, real sequence labels, and fail-closed stage control."""
import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithms.latent_imagination.world.data import (make_family,transition,branch_targets,
    SyntheticData,generate,canonical)
from algorithms.latent_imagination.world.branches import full_branch_loss_chunks,gates,evaluate
from algorithms.latent_imagination.world.objectives import supervised_world_loss
from algorithms.latent_imagination.world.model import WorldModel
from algorithms.latent_imagination.world import base as pipeline
from common.checkpoints import read_checkpoint,model_from_checkpoint
from common.training import setup


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    setup(13,'cpu')
    root=tmp_path_factory.mktemp('synthetic')
    generate(root/'data',families=180,horizon=10,seed=17,workers=1,max_rank=6)
    return root,SyntheticData(root/'data')


def tiny():
    return WorldModel(model_version=2,width=16,latent_dim=4,tokenizer_layers=1,dynamics_layers=1,heads=2)


def test_all_branch_probabilities_and_noop():
    state=np.array([0,1,0,0,2,3,4,5,3,4,5,6,4,5,6,7],dtype=np.uint8)
    after,p,boards,masks,dones=transition(state,0)
    assert np.allclose(p[[2,4,6]],[.3]*3)
    assert np.allclose(p[[3,5,7]],[1/30]*3)
    assert p[32]==0
    _,q,noops,_,_=transition(after,0)
    assert q[32]==1 and q[:32].sum()==0
    np.testing.assert_array_equal(noops[32],after)
    for c in np.flatnonzero(p):
        expected=after.copy();expected[c//2]=1+c%2
        np.testing.assert_array_equal(boards[c],expected)
    assert masks.shape==(33,4) and dones.shape==(33,)


def test_generator_split_isolation_truncation_and_hashes(synthetic):
    root,data=synthetic
    owners={}
    for split,part in data.splits.items():
        seq=part['sequence'];valid=seq['valid'].bool()
        # Short windows that survive horizon are not marked terminal.
        assert (~seq['dones'][:,-1] & valid[:,-1]).any()
        assert torch.equal(seq['states'][:,1:][valid[:,1:]],seq['next_states'][:,:-1][valid[:,1:]])
        assert not (seq['dones'][:,:-1] & valid[:,1:]).any()
        for values in [part['boards'],seq['states'][valid],seq['afterstates'][valid],seq['next_states'][valid]]:
            for board in values:
                key=canonical(board.numpy())
                assert owners.get(key,split)==split
                owners[key]=split
        assert torch.allclose(seq['chance_probs'][valid].sum(-1),torch.ones(int(valid.sum())))
    original=(root/'data'/'manifest.json').read_bytes()
    generate(root/'data',families=180,horizon=10,seed=17,workers=1,max_rank=6)
    assert (root/'data'/'manifest.json').read_bytes()==original
    with pytest.raises(ValueError,match='settings changed'):
        generate(root/'data',families=181,horizon=10,seed=17,workers=1,max_rank=6)


def test_full_distribution_loss_is_independent_of_observed_event(synthetic):
    _,data=synthetic;model=tiny();model.set_phase('world')
    batch=data.sample_sequence('train',4,1,'cpu',np.random.default_rng(1))
    _,terms=supervised_world_loss(model,batch)
    changed={k:v.clone() for k,v in batch.items()};changed['events']=(changed['events']+1)%33
    _,other=supervised_world_loss(model,changed)
    torch.testing.assert_close(terms['event_prior_nll'],other['event_prior_nll'],atol=0,rtol=0)
    with torch.no_grad():
        z=model.world.encode(batch['states'][:,0]);u=model.world.afterstate(z,batch['actions'][:,0])
        expected=-(batch['chance_probs'][:,0]*model.world.chance_prior(u,z,batch['actions'][:,0]).log_softmax(-1)).sum(-1).mean()
    torch.testing.assert_close(terms['event_prior_nll'],expected)


def test_chunked_full_branch_gradients_match_unchunked(synthetic):
    _,data=synthetic;setup(11,'cpu');model=tiny();model.set_phase('world')
    # Move zero-initialized transition readouts off identity to exercise both blocks.
    with torch.no_grad():
        model.world.action_transition.output.weight.normal_(std=.01)
        model.world.event_transition.output.weight.normal_(std=.01)
    clone=deepcopy(model)
    batch=data.sample('train',3,'cpu',np.random.default_rng(91))
    total=0
    for loss in full_branch_loss_chunks(model,batch,4): total+=float(loss.detach());loss.backward()
    other=sum(full_branch_loss_chunks(clone,batch,10000));other.backward()
    assert total==pytest.approx(float(other.detach()),rel=1e-5)
    for p,q in zip(model.parameters(),clone.parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=3e-4,rtol=2e-4)
    assert all(p.grad is None for p in model.world.tokenizer.parameters())
    assert model.world.event_embedding.weight.grad.abs().sum()>0


def test_gate_requires_depth_and_rejects_nonfinite():
    assert not gates('tokenizer',{'reconstruction':float('nan')})['passed']
    check=gates('world10',{})
    assert 'step_10_prefix' in check['failures']
    metrics={k:(1 if row['direction']=='min' else 0) for k,row in check['checks'].items()}
    assert gates('world10',metrics)['passed']
    metrics.pop('step_10_prefix')
    assert not gates('world10',metrics)['passed']


def test_failed_stage_retrains_and_never_advances(synthetic,tmp_path,monkeypatch):
    _,data=synthetic
    cfg={**pipeline.DEFAULTS,'width':16,'latent_dim':4,'layers':1,'heads':2,'device':'cpu',
         'batch_size':4,'validation_samples':4,'eval_every':1,'plot_every':1,'max_updates_per_stage':2}
    (tmp_path/'data').mkdir()
    pipeline.atomic_json(tmp_path/'data'/'verification.json',dict(passed=True,data_sha256=data.digest))
    # Synthetic validation responses test control flow, not world-model quality.
    monkeypatch.setattr(pipeline,'evaluate',lambda *a,**kw:{'reconstruction':0.})
    for name in ('training_plots','reconstruction_plot','progress_plot'): monkeypatch.setattr(pipeline,name,lambda *a,**kw:None)
    assert not pipeline.train_stage(tmp_path,cfg,data,'tokenizer')
    assert read_checkpoint(tmp_path/'tokenizer'/'last.pt')['iteration']==2
    assert not (tmp_path/'tokenizer'/'passed.pt').exists()
    with pytest.raises(ValueError,match='must pass'):
        pipeline.train_stage(tmp_path,cfg,data,'world1')
    assert not (tmp_path/'world1').exists()
    monkeypatch.setattr(pipeline,'evaluate',lambda *a,**kw:{'reconstruction':1.})
    cfg['max_updates_per_stage']=4
    assert pipeline.train_stage(tmp_path,cfg,data,'tokenizer')
    report=json.loads((tmp_path/'tokenizer'/'validation.json').read_text())
    assert report['consecutive_passes']==2 and report['iteration']==4
    assert pipeline.verify_predecessor(tmp_path,'world1',data)==tmp_path/'tokenizer'/'passed.pt'
    report['checkpoint_sha256']='wrong';pipeline.atomic_json(tmp_path/'tokenizer'/'validation.json',report)
    with pytest.raises(ValueError,match='does not match'): pipeline.verify_predecessor(tmp_path,'world1',data)


def test_audit_failure_forces_more_training(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(pipeline,'train_stage',lambda root,cfg,data,stage,minimum_updates=0: calls.append((stage,minimum_updates)) or True)
    results=iter([dict(passed=False,iteration=200),dict(passed=True,iteration=400)])
    monkeypatch.setattr(pipeline,'audit',lambda *a:next(results))
    assert pipeline.run(tmp_path,dict(eval_every=200),None)
    assert not (tmp_path/'behavior').exists() and not (tmp_path/'imagine').exists()
    assert calls[-1]==('world10',400)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_mps_full_branch_and_recurrent_training(synthetic):
    _,data=synthetic;model=tiny().to('mps');model.set_phase('world')
    rng=np.random.default_rng(9)
    batch=data.sample_sequence('train',2,10,'mps',rng)
    loss,_=supervised_world_loss(model,batch,data.sample_heads('train',4,'mps',rng));loss.backward()
    for value in full_branch_loss_chunks(model,data.sample('train',2,'mps',rng),8): value.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result=evaluate(model,data,'world3',samples=2,batch_size=2)
    assert 'prior_tv' in result and 'step_3_prefix' in result


def test_cpu_pipeline_resume_preserves_optimizer_and_rng(synthetic,tmp_path,monkeypatch):
    _,data=synthetic
    cfg={**pipeline.DEFAULTS,'width':16,'latent_dim':4,'layers':1,'heads':2,'device':'cpu',
         'batch_size':4,'validation_samples':4,'eval_every':1,'plot_every':1,'max_updates_per_stage':2}
    monkeypatch.setattr(pipeline,'evaluate',lambda *a,**kw:{'reconstruction':0.})
    for name in ('training_plots','reconstruction_plot','progress_plot'): monkeypatch.setattr(pipeline,name,lambda *a,**kw:None)
    whole,part=tmp_path/'whole',tmp_path/'part'
    for root in (whole,part):
        (root/'data').mkdir(parents=True)
        pipeline.atomic_json(root/'data'/'verification.json',dict(passed=True,data_sha256=data.digest))
    pipeline.train_stage(whole,cfg,data,'tokenizer')
    pipeline.train_stage(part,{**cfg,'max_updates_per_stage':1},data,'tokenizer')
    pipeline.train_stage(part,cfg,data,'tokenizer')
    a=read_checkpoint(whole/'tokenizer'/'last.pt');b=read_checkpoint(part/'tokenizer'/'last.pt')
    for key,value in a['model'].items(): torch.testing.assert_close(value,b['model'][key],atol=0,rtol=0)
    for key,row in a['optimizer']['state'].items():
        for name,value in row.items(): torch.testing.assert_close(value,b['optimizer']['state'][key][name],atol=0,rtol=0)


def test_whole_game_failure_is_in_audit_report_and_plot_gates(synthetic,tmp_path,monkeypatch):
    from common.checkpoints import save_checkpoint
    from algorithms.latent_imagination.runtime import file_digest
    _,data=synthetic
    directory=tmp_path/'world10';directory.mkdir()
    model=tiny()
    with torch.no_grad():
        model.world.terminal[-1].weight.zero_();model.world.terminal[-1].bias.fill_(10)
        model.world.legality[-1].weight.zero_();model.world.legality[-1].bias.fill_(-10)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    checkpoint=directory/'passed.pt'
    save_checkpoint(checkpoint,model,optimizer,2,'latent_dreamer',{},0)
    pipeline.atomic_json(directory/'validation.json',dict(passed=True,consecutive_passes=2,
        iteration=2,data_sha256=data.digest,checkpoint_sha256=file_digest(checkpoint)))
    # Pretend only the preceding short-rollout metrics passed. The actual game
    # is generated and checked with Board, so it must still fail independently.
    criteria=gates('world10',{})['checks']
    metrics={k:1. if r['direction']=='min' else 0. for k,r in criteria.items()}
    metrics['depth_samples']={}
    monkeypatch.setattr(pipeline,'evaluate',lambda *a,**kw:metrics)
    cfg={**pipeline.DEFAULTS,'device':'cpu','workers':1,'audit_games':1,'audit_max_steps':5}
    result=pipeline.audit(tmp_path,cfg,data)
    assert not result['passed'] and 'whole_games' in result['failures']
    assert not result['gates']['checks']['whole_games']['passed']
    assert (tmp_path/'audit'/'validation'/'update_00000002'/'validation.png').exists()
    assert (tmp_path/'audit'/'validation'/'update_00000002'/'game_4000000.html').exists()
