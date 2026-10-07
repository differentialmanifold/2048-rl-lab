"""Neural repair data, gradients, gating and continuation contracts."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_curriculum_v3 import curriculum, tiny
from test_latent_imagination import tiny_model
from algorithms.latent_imagination.world import trajectories as repair
from algorithms.latent_imagination.world.trajectory_data import label_step, ProbeData, policy_trace
from algorithms.latent_imagination.rollout import collect_games
from algorithms.latent_imagination.world.objectives import supervised_world_loss
from algorithms.latent_imagination.world.data import atomic_json
from algorithms.latent_imagination.world.branches import full_branch_loss_chunks
from algorithms.latent_imagination.world.verification import reference
from algorithms.latent_imagination.runtime import file_digest
from common.training import setup
from common.checkpoints import save_checkpoint, read_checkpoint


def probes_at(path,base,model):
    path.mkdir(parents=True)
    boards=base.heads['train']['boards'];boards=boards[~base.heads['train']['dones']][:90]
    with torch.no_grad():latents=model.world.encode(boards)
    splits={}
    for i,split in enumerate(('train','validation','test')):
        splits[split]=[dict(board=b,latent=z+.01,depth=500+j,hard=j%2==0,action=j%4,seed=i)
                       for j,(b,z) in enumerate(zip(boards[i*30:(i+1)*30],latents[i*30:(i+1)*30]))]
    torch.save(splits,path/'probes.pt')
    atomic_json(path/'manifest.json',dict(spec=dict(base_sha256=base.digest),sha256=file_digest(path/'probes.pt')))
    return ProbeData(path,base)


def test_teacher_labels_include_noop_and_complete_spawn_distribution():
    board=np.array([1,1,0,0]+[0]*12,dtype=np.uint8);rng=np.random.default_rng(4)
    row=label_step(board,0,rng)
    assert row['afterstates'].tolist()==[2]+[0]*15
    assert np.isclose(row['chance_probs'][0:32:2].sum(),.9)
    assert np.isclose(row['chance_probs'][1:32:2].sum(),.1)
    assert row['chance_probs'][32]==0 and row['chance_probs'][:2].sum()==0
    noop=label_step(np.array([1]+[0]*15,dtype=np.uint8),0,rng)
    assert noop['events']==32 and noop['rewards']==0 and noop['chance_probs'][32]==1


def test_shared_event_legal_is_learned_marginal_without_decoding(monkeypatch):
    from algorithms.latent_imagination.world.model import WorldModel
    model=WorldModel(model_version=2,width=16,latent_dim=4,tokenizer_layers=1,
        dynamics_layers=1,heads=2,state_head_version=3)
    model.set_phase('world');world=model.world
    monkeypatch.setattr(world,'decode',lambda *a:pytest.fail('No board decoder in legal head'))
    z=torch.randn(2,16,4)
    done,legal=world.state_heads(z)
    states=z[:,None].expand(-1,4,-1,-1).reshape(-1,16,4);actions=torch.arange(4).repeat(2)
    after=world.afterstate(states,actions);probs=world.event_probabilities(states,after,actions)
    torch.testing.assert_close(legal.sigmoid(),(1-probs[:,32]).reshape(2,4))
    legal.sum().backward()
    assert world.prior[0].weight.grad.abs().sum()>0
    assert not any(p.requires_grad for p in world.legality.parameters())


def test_repair_trace_matches_training_without_decoder_or_oracle(monkeypatch):
    setup(6,'cpu');model=tiny_model()
    result=collect_games(model,1,.999,113,max_steps=4)
    import algorithms.latent_imagination.world.trajectory_data as data
    def forbidden(*a,**kw):pytest.fail('Oracle/decoder must not drive latent audit generation')
    monkeypatch.setattr(data,'reference',forbidden);monkeypatch.setattr(model.world,'decode',forbidden)
    trace=policy_trace(model,113,4)
    assert [s['action'] for s in trace['steps']]==result[2].tolist()
    assert sum(float(s['reward']) for s in trace['steps'])==result[7][0]['spawn_return']
    assert trace['stop_reason']=='max_steps'


def test_rollout_reports_zero_reward_loops_without_using_board_rules(monkeypatch):
    setup(6,'cpu');model=tiny_model();world=model.world
    def output(u,event,nxt):
        batch=len(nxt)
        return (torch.tensor([[10.,-10.,-10.]]).expand(batch,-1),
                torch.full((batch,),-10.),torch.full((batch,4),10.))
    monkeypatch.setattr(world,'predict_outputs',output)
    result=collect_games(model,1,.999,113,max_steps=6)
    assert result[7][0]['zero_rewards']==6
    assert result[7][0]['longest_zero_reward_streak']==6


@pytest.mark.parametrize('device',['cpu',pytest.param('mps',marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(),reason='MPS unavailable'))])
def test_long_latent_supervision_reaches_heads_prior_and_dynamics(curriculum,tmp_path,device):
    _,base=curriculum;setup(9,'cpu');model=tiny();probes=probes_at(tmp_path/'probes',base,model)
    model.to(device);model.set_phase('world');initial=deepcopy(model.world.tokenizer.state_dict())
    batch,z=probes.sample('train',8,3,device,np.random.default_rng(20))
    assert batch['rewards'].dtype==torch.float32 and z.shape[1:]==(16,4)
    for i in range(len(batch['states'])):
        for t in range(3):
            if not batch['valid'][i,t]:continue
            truth=reference(batch['states'][i,t].cpu().numpy(),int(batch['actions'][i,t]))
            assert batch['masks'][i,t].tolist()==truth['mask']
            assert batch['afterstates'][i,t].tolist()==truth['afterstate']
    loss,metrics=supervised_world_loss(model,batch,enhanced=True,initial_latent=z,
        consistency_weight=4,head_margin=2,invalid_weight=20)
    loss.backward();assert torch.isfinite(loss) and 'recurrent_root_head_nll' in metrics
    for module in (model.world.legality,model.world.prior,model.world.action_transition,model.world.event_transition):
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters())
    assert all(p.grad is None for p in model.world.tokenizer.parameters())
    torch.optim.Adam([p for p in model.parameters() if p.requires_grad]).step()
    for k,v in initial.items():torch.testing.assert_close(v,model.world.tokenizer.state_dict()[k],rtol=0,atol=0)


def test_recurrent_full_branch_gradients_and_fail_closed_gate(curriculum,tmp_path):
    _,base=curriculum;model=tiny();model.set_phase('world');probes=probes_at(tmp_path/'probes',base,model)
    batch,z=probes.sample_branches(2,'cpu',np.random.default_rng(6))
    for loss in full_branch_loss_chunks(model,batch,8,enhanced=True,initial_latent=z):loss.backward()
    assert model.world.prior[0].weight.grad.abs().sum()>0
    metrics=dict(recurrent_legal=1.,encoded_legal=1.,readout_agreement=1.,illegal_false_positive=0.)
    games=[dict(role=role,passed=True) for role in ('policy','random','regression')]
    assert repair.trajectory_gates(dict(passed=True),metrics,games)['passed']
    games[-1]['passed']=False
    assert not repair.trajectory_gates(dict(passed=True),metrics,games)['passed']
    with pytest.raises(ValueError,match='Missing long-policy'):
        repair.require_trajectory_report(tmp_path,dict(world_repair_version=1))


def test_pipeline_world_failure_does_not_report_success(monkeypatch):
    from algorithms.latent_imagination import pipeline
    from algorithms.latent_imagination.world import pipeline as world
    monkeypatch.setattr(world,'main',lambda argv:False)
    with pytest.raises(SystemExit) as error:pipeline.main(['world','--run-dir','unused'])
    assert error.value.code==2


@pytest.mark.parametrize('stage_folder', ['repair', 'trajectories'])
def test_repair_resume_preserves_optimizer_and_only_publishes_after_gate(curriculum,tmp_path,monkeypatch,stage_folder):
    # Synthetic passing reports test pipeline control/provenance, not accuracy.
    _,base=curriculum;setup(42,'cpu');initial=tiny()
    monkeypatch.setattr(repair,'evaluate',lambda *a,**k:{'fixture':1.})
    monkeypatch.setattr(repair,'gates',lambda *a,**k:dict(passed=True,checks={},failures=[]))
    monkeypatch.setattr(repair,'probe_metrics',lambda *a,**k:dict(recurrent_legal=1.,encoded_legal=1.,
        readout_agreement=1.,illegal_false_positive=0.,samples=1))
    monkeypatch.setattr(repair,'long_audit',lambda *a,**k:[dict(role=r,passed=True) for r in ('policy','random','regression')])
    monkeypatch.setattr(repair,'plots',lambda *a:None)
    def prepare_round(root,cfg,data,round_index,world_path):
        assert read_checkpoint(world_path)['iteration']==round_index
        return probes_at(root/stage_folder/f'probes_{round_index:04d}',data,initial)
    monkeypatch.setattr(repair,'prepare',prepare_round)
    cfg={**repair.DEFAULTS,'device':'cpu','workers':1,'updates':3,'max_updates':3,
         'batch_size':2,'horizon':2,'eval_every':1,'plot_every':1,'refresh_every':1}
    roots=[tmp_path/'whole',tmp_path/'part'];datasets=[];probes=[]
    for root in roots:
        (root/stage_folder).mkdir(parents=True)
        save_checkpoint(root/'source.pt',initial,torch.optim.Adam(initial.parameters()),0,'latent_dreamer',{},0.)
        teacher=tiny_model();save_checkpoint(root/'teacher.pt',teacher,torch.optim.Adam(teacher.policy.parameters()),1,'latent_afterstate_ppo',{},0.)
        data=deepcopy(base);datasets.append(data)
        probes.append(probes_at(root/stage_folder/'probes_0000',data,initial))
    assert repair.run(roots[0],cfg,datasets[0],probes[0])
    assert not repair.run(roots[1],{**cfg,'max_updates':1},datasets[1],probes[1])
    assert not (roots[1]/'world10/passed.pt').exists()
    assert repair.run(roots[1],cfg,datasets[1],probes[1])
    a,b=[read_checkpoint(r/stage_folder/'last.pt') for r in roots]
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    for i,state in a['optimizer']['state'].items():
        for key,v in state.items():torch.testing.assert_close(v,b['optimizer']['state'][i][key],atol=0,rtol=0)
    assert repair.require_trajectory_report(roots[1],b)['passed']
    before=(roots[1]/stage_folder/'last.pt').read_bytes()
    assert repair.run(roots[1],cfg,datasets[1],probes[1])
    assert before==(roots[1]/stage_folder/'last.pt').read_bytes()
    # Tampering with supervised labels invalidates policy admission.
    (roots[1]/stage_folder/'probes_0002/probes.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):repair.require_trajectory_report(roots[1],b)
