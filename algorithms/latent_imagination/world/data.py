"""Synthetic short sequences and full chance labels; oracle is training-only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common.parallel import GamePool
from ..runtime import file_digest
from .verification import reference

KINDS = ('sparse', 'dense', 'merge', 'blocked', 'large', 'terminal', 'reachable')
SPLITS = ('train', 'validation', 'test')


def canonical(board):
    board = np.asarray(board, dtype=np.uint8).reshape(4, 4)
    return min(x.tobytes() for k in range(4) for x in (np.rot90(board,k), np.rot90(board,k).T))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def branch_targets(after, probabilities):
    """Expand stored teacher labels, never used by neural inference."""
    result = after[..., None, :].expand(*after.shape[:-1], 33, 16).clone()
    for c in range(32):
        result[..., c, c//2] = 1+c%2
    return result


def transition(state, action):
    truth = reference(state, action)
    after = np.asarray(truth['afterstate'], dtype=np.uint8)
    probs = np.zeros(33, dtype=np.float32)
    if truth['changed']:
        empty = np.flatnonzero(after == 0)
        probs[2*empty] = (1-truth['spawn4_probability'])/len(empty)
        probs[2*empty+1] = truth['spawn4_probability']/len(empty)
    else:
        probs[32] = 1
    boards = np.repeat(after[None], 33, axis=0)
    masks = np.zeros((33,4),dtype=bool)
    dones = np.zeros(33,dtype=bool)
    for c in np.flatnonzero(probs):
        if c < 32:
            boards[c,c//2] = 1+c%2
        target = reference(boards[c])
        masks[c], dones[c] = target['mask'], target['done']
    return after, probs, boards, masks, dones


def construct(rng, kind, max_rank):
    if kind == 'reachable':
        state = np.zeros(16,dtype=np.uint8)
        state[rng.choice(16,2,replace=False)] = rng.choice([1,2],2,p=[.9,.1])
        for _ in range(int(rng.integers(20,160))):
            mask = reference(state)['mask']
            if not any(mask):
                break
            _,p,boards,_,_ = transition(state,int(rng.choice(np.flatnonzero(mask))))
            state = boards[rng.choice(33,p=p.astype(float)/p.astype(float).sum())].copy()
        return state
    high = max_rank if kind in ('large','terminal') else min(max_rank,7)
    state = rng.integers(1,high+1,16,dtype=np.uint8)
    if kind == 'terminal':
        # Varied no-merge boards, not a single repeated checkerboard.
        grid = state.reshape(4,4)
        for r in range(4):
            for c in range(4):
                candidates = [v for v in range(1,high+1)
                              if (r==0 or v!=grid[r-1,c]) and (c==0 or v!=grid[r,c-1])]
                grid[r,c] = rng.choice(candidates)
    else:
        empty = int(rng.integers(7,13) if kind=='sparse' else rng.integers(0,5))
        state[rng.choice(16,empty,replace=False)] = 0
        if kind in ('merge','blocked'):
            rank = int(rng.integers(1,high+1))
            row = int(rng.integers(4))
            state[row*4:row*4+4] = ([rank]*4 if kind=='merge' else [rank,0,rank,min(high,rank+1)])
    grid = np.rot90(state.reshape(4,4),int(rng.integers(4)))
    if rng.random()<.5:
        grid = grid.T
    return grid.copy().reshape(-1)


def make_family(job):
    index, seed, horizon, max_rank = job
    rng = np.random.default_rng(seed+index*104729)
    kind = KINDS[index % len(KINDS)]
    root = construct(rng,kind,max_rank)
    bucket = int.from_bytes(hashlib.sha256(canonical(root)).digest()[:8],'big') % 10
    split = 'validation' if bucket == 0 else ('test' if bucket == 1 else 'train')
    info = reference(root)
    arrays = dict(states=np.zeros((4,horizon,16),np.uint8), afterstates=np.zeros((4,horizon,16),np.uint8),
        next_states=np.zeros((4,horizon,16),np.uint8), masks=np.zeros((4,horizon,4),bool),
        next_masks=np.zeros((4,horizon,4),bool), actions=np.zeros((4,horizon),np.int64),
        events=np.full((4,horizon),32,np.int64), rewards=np.zeros((4,horizon),np.float32),
        dones=np.zeros((4,horizon),bool), valid=np.zeros((4,horizon),np.float32),
        chance_probs=np.zeros((4,horizon,33),np.float32),
        branch_masks=np.zeros((4,horizon,33,4),bool), branch_dones=np.zeros((4,horizon,33),bool))
    for first in range(4):
        state = root.copy()
        for t in range(horizon):
            current = reference(state)
            if current['done']:
                break
            action = first if t==0 else int(rng.choice(4) if rng.random()<.2 else rng.choice(np.flatnonzero(current['mask'])))
            after,p,boards,masks,dones = transition(state,action)
            event = int(rng.choice(33,p=p.astype(float)/p.astype(float).sum()))
            nxt = boards[event]
            if int(max(after.max(),nxt.max())) >= 32:
                raise ValueError('Synthetic transition exceeds the model tile vocabulary')
            values = dict(states=state,afterstates=after,next_states=nxt,masks=current['mask'],
                next_masks=masks[event],actions=action,events=event,rewards=0 if event==32 else 2+2*(event%2),
                dones=dones[event],valid=1,chance_probs=p,branch_masks=masks,branch_dones=dones)
            for key,value in values.items():
                arrays[key][first,t] = value
            state = nxt.copy()
    # Siblings, all actions and their continuations form one indivisible family.
    return dict(index=index,kind=kind,split=split,root=torch.from_numpy(root.copy()),
                root_mask=torch.tensor(info['mask']),root_done=info['done'],
                sequence={k:torch.from_numpy(v) for k,v in arrays.items()})


def family_keys(family):
    """Disallow observed state/afterstate collisions across split families."""
    seq = family['sequence']; valid = seq['valid'].bool()
    boards = torch.cat([family['root'][None],seq['states'][valid],seq['afterstates'][valid],seq['next_states'][valid]])
    return {canonical(b.numpy()) for b in boards}


def generate(directory, families=6000, horizon=10, seed=0, workers=8, max_rank=12, progress=None):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    config = dict(format='synthetic2048_v1',families=families,horizon=horizon,seed=seed,max_rank=max_rank)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config:
            raise ValueError('Dataset settings changed; choose a new run directory')
        for item in manifest['shards']:
            if file_digest(directory/item['file']) != item['sha256']:
                raise ValueError('Synthetic shard checksum mismatch')
        if manifest['complete']:
            return manifest
    else:
        manifest = dict(config=config,shards=[],complete=False,next_candidate=0,rejected_overlap=0)
    owners, roots = {}, set()
    accepted = 0
    for item in manifest['shards']:
        for family in torch.load(directory/item['file'],weights_only=True):
            roots.add(canonical(family['root'].numpy()))
            for key in family_keys(family): owners[key]=family['split']
            accepted += 1
    with GamePool(workers) as pool:
        while accepted < families:
            start = manifest['next_candidate']
            jobs = [(i,seed,horizon,max_rank) for i in range(start,start+min(32,families-accepted))]
            chunk=[]
            for family in pool.map(make_family,jobs):
                keys=family_keys(family); root=canonical(family['root'].numpy())
                if root in roots or any(owners.get(k,family['split'])!=family['split'] for k in keys):
                    manifest['rejected_overlap'] += 1
                    continue
                roots.add(root)
                for key in keys: owners[key]=family['split']
                chunk.append(family); accepted+=1
            manifest['next_candidate']=start+len(jobs)
            if chunk:
                name=f'shard_{len(manifest["shards"]):05d}.pt'
                temp=directory/(name+'.tmp');torch.save(chunk,temp);temp.replace(directory/name)
                manifest['shards'].append(dict(file=name,sha256=file_digest(directory/name),families=len(chunk)))
            atomic_json(manifest_path,manifest)
            if progress: progress(accepted,families)
            if manifest['next_candidate'] > families*100:
                raise RuntimeError('Cannot construct enough isolated families in this range')
    manifest['complete']=True
    atomic_json(manifest_path,manifest)
    return manifest


class SyntheticData:
    def __init__(self,directory):
        self.directory=Path(directory)
        manifest=json.loads((self.directory/'manifest.json').read_text())
        if not manifest['complete']: raise ValueError('Dataset generation is incomplete')
        self.digest=file_digest(self.directory/'manifest.json'); self.horizon=manifest['config']['horizon']
        families=[]
        for shard in manifest['shards']:
            path=self.directory/shard['file']
            if file_digest(path)!=shard['sha256']: raise ValueError('Dataset shard changed')
            families.extend(torch.load(path,weights_only=True))
        self.splits={};self.coverage={}
        for split in SPLITS:
            group=[f for f in families if f['split']==split]
            if not group: raise ValueError(f'No {split} families; generate more families')
            seq={k:torch.cat([f['sequence'][k] for f in group]) for k in group[0]['sequence']}
            keep=seq['valid'][:,0].bool()
            seq={k:v[keep] for k,v in seq.items()}
            if not keep.any(): raise ValueError(f'No live sequences in {split}')
            flat={k:v[seq['valid'].bool()] for k,v in seq.items()}
            boards=torch.stack([f['root'] for f in group])
            masks=torch.stack([f['root_mask'] for f in group])
            dones=torch.tensor([f['root_done'] for f in group])
            self.splits[split]=dict(sequence=seq,flat=flat,boards=boards,masks=masks,dones=dones)
            self.coverage[split]=dict(families=len(group),sequences=int(keep.sum()),transitions=len(flat['actions']),
                kinds={k:sum(f['kind']==k for f in group) for k in KINDS},
                actions=torch.bincount(flat['actions'],minlength=4).tolist(),
                terminal_roots=int(dones.sum()),invalid_actions=int((flat['events']==32).sum()),
                empty_cells=torch.bincount((boards==0).sum(-1),minlength=17).tolist(),
                tile_ranks=torch.bincount(boards.flatten().long(),minlength=32).tolist(),
                probability_mass=flat['chance_probs'].sum(0).tolist())

    @staticmethod
    def take(part,count,device,rng):
        ids=torch.from_numpy(rng.integers(len(next(iter(part.values()))),size=count))
        return {k:v[ids].to(device) for k,v in part.items()}

    def sample_sequence(self,split,count,steps,device,rng):
        if steps>self.horizon: raise ValueError('Sequence length exceeds generated data horizon')
        sample=self.take(self.splits[split]['sequence'],count,device,rng)
        return {k:v[:,:steps] for k,v in sample.items()}

    def sample(self,split,count,device,rng):
        return self.take(self.splits[split]['flat'],count,device,rng)

    def sample_heads(self,split,count,device,rng):
        part=self.splits[split]; done=part['dones']
        positive=done.nonzero().flatten();negative=(~done).nonzero().flatten()
        if not len(positive) or not len(negative): raise ValueError('Need terminal and live root examples in every split')
        n=count//2
        ids=torch.cat((positive[rng.integers(len(positive),size=n)],negative[rng.integers(len(negative),size=count-n)]))
        frequency=float(done.float().mean())
        weights=torch.cat((torch.full((n,),frequency/(n/count)),torch.full((count-n,),(1-frequency)/((count-n)/count))))
        return dict(states=part['boards'][ids].to(device),masks=part['masks'][ids].to(device),
                    dones=done[ids].to(device),weights=weights.to(device))

    def tokenizer_boards(self,split,count,device,rng):
        batch=self.sample(split,count,device,rng)
        heads=self.sample_heads(split,count,device,rng)
        targets=branch_targets(batch['afterstates'],batch['chance_probs'])
        # Uniform supported branches helps rare 4 outcomes, without changing P's labels.
        support=batch['chance_probs'].cpu().numpy()>0
        events=[rng.choice(np.flatnonzero(row)) for row in support]
        branch=targets[torch.arange(count,device=device),torch.tensor(events,device=device)]
        return torch.cat((heads['states'],batch['states'],batch['afterstates'],batch['next_states'],branch))
