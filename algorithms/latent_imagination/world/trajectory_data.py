"""Offline labels on long recurrent latents; no oracle enters policy inference."""
import hashlib
from pathlib import Path
import json

import numpy as np
import torch

from common.checkpoints import read_checkpoint, model_from_checkpoint
from common.models import masked_categorical
from common.parallel import GamePool
from common.training import setup
from ..rollout import initial_boards, draw
from ..runtime import file_digest
from .data import canonical, atomic_json
from .verification import reference
from .boundaries import canonical_keys


@torch.no_grad()
def policy_trace(model, seed, max_steps=10000, random_policy=False):
    """Same CPU RNGs/batch shape as a single training worker, no decoder/rules."""
    world=model.world;device=next(world.parameters()).device
    initial=initial_boards([seed]).to(device);z=world.encode(initial)
    action_rng=[np.random.default_rng(seed+20000000)]
    chance_rng=[np.random.default_rng(seed+30000000)]
    trace=dict(initial_ranks=initial[0].long().cpu().tolist(),initial_z=z.cpu(),seed=seed,
               steps=[],stop_reason='max_steps')
    terminal,legal=world.state_heads(z)
    for _ in range(max_steps):
        mask=legal>=0
        if bool((terminal>=0).item()) or not bool(mask.any()):
            break
        candidates=model.candidate_latents(z)
        logits=(torch.zeros(1,4,device=device) if random_policy else model.policy(candidates)[0])
        action=draw(masked_categorical(logits,mask).probs,action_rng,[0])
        u=candidates[torch.arange(1,device=device),action]
        probabilities=world.event_probabilities(z,u,action)
        event=draw(probabilities,chance_rng,[0]);nxt=world.event_step(u,event)
        reward_logits,done,next_legal=world.predict_outputs(u,event,nxt)
        if not all(bool(torch.isfinite(t).all()) for t in (u,probabilities,nxt,reward_logits,done,next_legal)):
            trace['stop_reason']='non_finite_prediction';break
        trace['steps'].append(dict(action=int(action.item()),terminal=terminal.cpu(),legal=legal.cpu(),
            afterstate=u.cpu(),probabilities=probabilities.cpu(),event=event.cpu(),next_z=nxt.cpu(),
            reward=world.reward_values[reward_logits.argmax(-1)].cpu(),
            reward_mean=(reward_logits.softmax(-1)*world.reward_values).sum(-1).cpu(),
            next_terminal=done.cpu(),next_legal=next_legal.cpu()))
        z,terminal,legal=nxt,done,next_legal
    trace.update(final_terminal=terminal.cpu(),final_legal=legal.cpu())
    if bool((terminal>=0).item()):trace['stop_reason']='model_terminal'
    elif not bool((legal>=0).any()):trace['stop_reason']='no_predicted_legal_actions'
    return trace


def probe_job(job):
    policy_path, world_path, seed, max_steps, stride, random_policy = job
    setup(seed,'cpu');model=model_from_checkpoint(read_checkpoint(policy_path)).eval()
    if world_path:
        world=model_from_checkpoint(read_checkpoint(world_path)).world
        model.world=world
        model.world.requires_grad_(False);model.world.eval()
    trace=policy_trace(model,seed,max_steps,random_policy)
    latents=[trace['initial_z'],*[s['next_z'] for s in trace['steps']]]
    boards=[]
    with torch.no_grad():
        for start in range(0,len(latents),128):
            boards.extend(model.world.decode(torch.cat(latents[start:start+128])).argmax(-1).cpu())
    records=[];seen={}
    for depth,(z,board) in enumerate(zip(latents,boards)):
        truth=reference(board.numpy())
        logits=trace['steps'][depth]['legal'] if depth<len(trace['steps']) else trace['final_legal']
        hard=(logits>=0).reshape(-1).tolist()!=truth['mask']
        hint=trace['steps'][depth]['action'] if depth<len(trace['steps']) else 0
        key=board.numpy().tobytes()
        # Do not let a 9,000-step stationary loop monopolize the repair corpus.
        if (depth%stride and not hard) or seen.get(key,0)>=4:continue
        seen[key]=seen.get(key,0)+1
        records.append(dict(latent=z[0].clone(),board=board.to(torch.uint8),depth=depth,
                            hard=hard,action=hint,seed=seed))
    return records,dict(seed=seed,steps=len(trace['steps']),stop_reason=trace['stop_reason'],probes=len(records))


def prepare_probes(directory,policy_path,world_path,base_data,episodes=32,seed=0,workers=8,
                   max_steps=10000,stride=8,regression_seeds=(),round_index=0):
    """Episode-separated discovery, then board/D4 separation from base and peers.

    Labels are constructed only for train roots. Validation/test roots whose
    semantics overlap training are discarded; original held-out boards cannot
    enter training, including sampled afterstates/next boards (checked later).
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'probes.pt';manifest_path=directory/'manifest.json'
    spec=dict(version=1,policy_sha256=file_digest(policy_path),world_sha256=file_digest(world_path),
        base_sha256=base_data.digest,episodes=episodes,seed=seed,max_steps=max_steps,stride=stride,
        regression_seeds=list(regression_seeds),round_index=round_index)
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['spec']!=spec or file_digest(path)!=old['sha256']:raise ValueError('Trajectory corpus changed')
        return ProbeData(directory,base_data)
    splits={'train':[],'validation':[],'test':[]};summaries=[]
    # Distinct seed namespaces; audit seeds are separate again and never fitted.
    jobs=[]
    for split,offset,count in [('train',6100000,episodes),('validation',7100000,max(2,episodes//4)),
                                ('test',8100000,max(2,episodes//4))]:
        seeds=list(range(offset+seed+round_index*10000,offset+seed+round_index*10000+count))
        if split=='train':seeds=list(regression_seeds)+seeds
        jobs.extend((split,(str(policy_path),str(world_path),s,max_steps,stride,i%5==4)) for i,s in enumerate(seeds))
    with GamePool(workers) as pool:
        # Bounded groups keep traces and pickled tensors from accumulating.
        for start in range(0,len(jobs),max(1,workers)):
            group=jobs[start:start+max(1,workers)]
            for (split,_),(records,summary) in zip(group,pool.map(probe_job,[j for _,j in group])):
                splits[split].extend(records);summaries.append(dict(split=split,**summary))
                print(json.dumps(dict(event='trajectory_probes',split=split,**summary)),flush=True)
    owner={k:'train' for k in canonical_keys(base_data.heads['train']['boards'].numpy())}
    for part in base_data.splits.values():
        # Base held-out keys below override any training labels here.
        for key in ('states','afterstates','next_states'):
            owner.update({k:'train' for k in canonical_keys(part['flat'][key].numpy())})
    owner.update({k:'base_holdout' for k in base_data.forbidden})
    for split in splits:
        retained=[]
        for row in splits[split]:
            key=canonical(row['board'].numpy())
            if owner.get(key,split)!=split:continue
            owner[key]=split;retained.append(row)
        splits[split]=retained
        if not retained:raise ValueError(f'No isolated {split} probes; collect more episodes')
    temp=path.with_suffix('.tmp');torch.save(splits,temp);temp.replace(path)
    atomic_json(manifest_path,dict(spec=spec,sha256=file_digest(path),games=summaries,
        counts={k:len(v) for k,v in splits.items()}))
    return ProbeData(directory,base_data)


def label_step(state,action,rng):
    truth=reference(state,action);after=np.asarray(truth['afterstate'],dtype=np.uint8)
    q=np.zeros(33,np.float32)
    if truth['changed']:
        empty=np.flatnonzero(after==0)
        q[2*empty]=(1-truth['spawn4_probability'])/len(empty)
        q[2*empty+1]=truth['spawn4_probability']/len(empty)
    else:q[32]=1
    event=int(rng.choice(33,p=q.astype(float)/q.astype(float).sum()))
    nxt=after.copy()
    if event<32:nxt[event//2]=1+event%2
    following=reference(nxt)
    return dict(states=np.asarray(state,dtype=np.uint8),afterstates=after,next_states=nxt,
        masks=np.array(truth['mask']),next_masks=np.array(following['mask']),actions=action,events=event,
        rewards=float(0 if event==32 else 2+2*(event%2)),dones=following['done'],valid=1.,chance_probs=q)


class ProbeData:
    def __init__(self,directory,base_data):
        self.directory=Path(directory);self.base=base_data
        manifest=json.loads((self.directory/'manifest.json').read_text())
        if manifest['spec']['base_sha256']!=base_data.digest or file_digest(self.directory/'probes.pt')!=manifest['sha256']:
            raise ValueError('Trajectory corpus provenance mismatch')
        self.splits=torch.load(self.directory/'probes.pt',weights_only=True)
        self.digest=file_digest(self.directory/'manifest.json')
        self.hard={s:np.array([i for i,r in enumerate(rows) if r['hard']],dtype=np.int64) for s,rows in self.splits.items()}
        self.forbidden=set(base_data.forbidden)
        self.forbidden.update(canonical(r['board'].numpy()) for s in ('validation','test') for r in self.splits[s])

    def sample(self,split,count,steps,device,rng):
        rows=self.splits[split];indices=rng.integers(len(rows),size=count)
        if split=='train' and len(self.hard[split]):
            indices[:count//2]=rng.choice(self.hard[split],count//2)
        sequences=[];latents=[]
        for i in indices:
            root=rows[int(i)];state=root['board'].numpy().copy();sequence=[]
            for t in range(steps):
                truth=reference(state)
                action=(root['action'] if t==0 and rng.random()<.5 else int(rng.integers(4)))
                row=label_step(state,action,rng)
                if split=='train' and any(canonical(row[k]) in self.forbidden for k in ('states','afterstates','next_states')):
                    if t==0:break
                    row['valid']=0.
                if truth['done'] or (sequence and not sequence[-1]['valid']):row['valid']=0.
                sequence.append(row);state=row['next_states']
            if len(sequence)!=steps:continue
            sequences.append(sequence);latents.append(root['latent'])
        if not sequences:raise ValueError('No isolated repair sequence sampled')
        batch={k:torch.as_tensor(np.array([[row[k] for row in sequence] for sequence in sequences]),device=device,
                    dtype=torch.float32 if k in ('valid','rewards','chance_probs') else None)
               for k in sequences[0][0]}
        # Explicit float32 keeps MPS compatible, including generated scalar labels.
        for k in ('valid','rewards','chance_probs'):batch[k]=batch[k].float()
        return batch,torch.stack(latents).to(device)

    def sample_branches(self,count,device,rng):
        from .data import transition
        sequence,z=self.sample('train',count,1,device,rng)
        batch={k:v[:,0] for k,v in sequence.items()}
        masks=[];dones=[]
        for state,action in zip(batch['states'].cpu().numpy(),batch['actions'].cpu().tolist()):
            _,_,_,m,d=transition(state,action);masks.append(m);dones.append(d)
        batch['branch_masks']=torch.as_tensor(np.array(masks),device=device)
        batch['branch_dones']=torch.as_tensor(np.array(dones),device=device)
        return batch,z
