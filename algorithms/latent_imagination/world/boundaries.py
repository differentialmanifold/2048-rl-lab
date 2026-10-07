"""Training-only boundary examples, isolated splits and random recurrent windows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common.parallel import GamePool
from ..runtime import file_digest
from .data import SyntheticData,atomic_json,canonical,construct
from .verification import reference


def canonical_keys(boards):
    """Vectorized equivalent of canonical(), retaining trailing zero bytes."""
    a=np.asarray(boards,dtype=np.uint8).reshape(-1,4,4)
    views=[v.reshape(-1,16) for k in range(4) for v in
           (np.rot90(a,k,axes=(1,2)),np.rot90(a,k,axes=(1,2)).transpose(0,2,1))]
    packed=np.ascontiguousarray(np.stack(views,axis=1)).view('S16').reshape(-1,8)
    return [bytes(v).ljust(16,b'\0') for v in np.sort(packed,axis=1)[:,0]]


def base_owners(data):
    owners={}
    for split,part in data.splits.items():
        boards=torch.cat([part['boards'],*[part['flat'][k] for k in ('states','afterstates','next_states')]])
        for start in range(0,len(boards),8192):
            for key in canonical_keys(boards[start:start+8192].numpy()):
                if key in owners and owners[key]!=split: raise ValueError('Base dataset split collision')
                owners[key]=split
    return owners


def boundary_family(job):
    index,seed,max_rank=job;rng=np.random.default_rng(seed+104729*index)
    high=int(rng.choice([3,min(6,max_rank),max_rank]))
    terminal=construct(rng,'terminal',high)
    neighbor=terminal.copy();kind=index%4
    if kind==0:
        row,col=int(rng.integers(4)),int(rng.integers(3))
        neighbor[row*4+col+1]=neighbor[row*4+col]
    elif kind==1:
        neighbor[int(rng.integers(16))]=0
    elif kind==2:
        neighbor[:]=0
        values=terminal.reshape(4,4)[:,0]  # Adjacent ranks already differ.
        neighbor.reshape(4,4)[:,0]=values
        neighbor=np.rot90(neighbor.reshape(4,4),int(rng.integers(4))).copy().reshape(16)
    else:
        neighbor=construct(rng,'blocked',high)
    bucket=int.from_bytes(hashlib.sha256(canonical(terminal)).digest()[:8],'big')%10
    split='validation' if bucket==0 else ('test' if bucket==1 else 'train')
    boards=np.stack([terminal,neighbor]);truth=[reference(b) for b in boards]
    return dict(index=index,split=split,kind=('merge_pair','one_empty','single_direction','blocked')[kind],
        boards=torch.from_numpy(boards),masks=torch.tensor([t['mask'] for t in truth]),
        dones=torch.tensor([t['done'] for t in truth]))


def prepare_boundaries(directory,data,families=12000,seed=0,max_rank=12,workers=8,progress=None):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    path=directory/'manifest.json'
    spec=dict(version=1,base_sha256=data.digest,families=families,seed=seed,max_rank=max_rank)
    manifest=json.loads(path.read_text()) if path.exists() else dict(spec=spec,shards=[],next_candidate=0,complete=False)
    if manifest['spec']!=spec: raise ValueError('Boundary dataset settings changed; use a new run directory')
    owners=base_owners(data);accepted=0
    for item in manifest['shards']:
        shard=directory/item['file']
        if file_digest(shard)!=item['sha256']: raise ValueError('Boundary shard checksum mismatch')
        for family in torch.load(shard,weights_only=True):
            for key in canonical_keys(family['boards'].numpy()): owners[key]=family['split']
            accepted+=1
    if manifest['complete']: return owners
    with GamePool(workers) as pool:
        while accepted<families:
            start=manifest['next_candidate'];size=min(256,families-accepted)
            chunk=[]
            for family in pool.map(boundary_family,[(i,seed+61000000,max_rank) for i in range(start,start+size)]):
                keys=canonical_keys(family['boards'].numpy());split=family['split']
                if any(key in owners for key in keys): continue
                for key in keys: owners[key]=split
                chunk.append(family);accepted+=1
            manifest['next_candidate']=start+size
            if chunk:
                name=f'shard_{len(manifest["shards"]):05d}.pt';tmp=directory/(name+'.tmp')
                torch.save(chunk,tmp);tmp.replace(directory/name)
                manifest['shards'].append(dict(file=name,sha256=file_digest(directory/name),families=len(chunk)))
            atomic_json(path,manifest)
            if progress: progress(accepted,families)
            if manifest['next_candidate']>families*100: raise RuntimeError('Unable to generate isolated boundary families')
    manifest['complete']=True;atomic_json(path,manifest)
    return owners


class CurriculumData:
    optimized=True

    def __init__(self,base,directory,owners=None):
        self.base=base;self.splits=base.splits;self.coverage=base.coverage;self.horizon=base.horizon
        self.directory=base.directory;self.base_digest=base.digest
        directory=Path(directory);path=directory/'manifest.json';manifest=json.loads(path.read_text())
        if not manifest['complete'] or manifest['spec']['base_sha256']!=base.digest:
            raise ValueError('Boundary corpus incomplete or bound to different base data')
        self.max_rank=manifest['spec']['max_rank'];families=[]
        for item in manifest['shards']:
            if file_digest(directory/item['file'])!=item['sha256']: raise ValueError('Boundary shard changed')
            families.extend(torch.load(directory/item['file'],weights_only=True))
        self.digest=hashlib.sha256((base.digest+file_digest(path)).encode()).hexdigest()
        owners=base_owners(base) if owners is None else owners
        for family in families:
            for key in canonical_keys(family['boards'].numpy()):
                if key in owners and owners[key]!=family['split']: raise ValueError('Boundary split collision')
                owners[key]=family['split']
        self.forbidden={k for k,v in owners.items() if v!='train'}
        self.heads={};self.indices={};self.coverage=dict(base.coverage)
        for split,part in self.splits.items():
            group=[f for f in families if f['split']==split]
            if not group: raise ValueError(f'Missing {split} boundary examples; generate more families')
            self.heads[split]={k:torch.cat([part[base_key],*[f[k] for f in group]])
                for k,base_key in [('boards','boards'),('masks','masks'),('dones','dones')]}
            flat=part['flat'];seq=part['sequence'];valid=seq['valid'].bool()
            self.indices[split]=dict(noop=(flat['events']==32).nonzero().flatten(),
                changed=(flat['events']!=32).nonzero().flatten(),
                terminal=flat['dones'].nonzero().flatten(),live=(~flat['dones']).nonzero().flatten(),
                starts=valid.nonzero(),lengths=valid.sum(-1))
            self.coverage[split]={**base.coverage[split],'boundary_families':len(group),
                'head_boards':len(self.heads[split]['boards'])}

    def sample(self,split,count,device,rng): return self.base.sample(split,count,device,rng)

    def sample_prior(self,split,count,device,rng):
        """Oversample no-ops; every individual conditional q stays unchanged."""
        ids=self.indices[split]
        n=max(1,count//4) if count>1 else int(rng.random()<.25)
        if not len(ids['noop']) or not len(ids['changed']): return self.sample(split,count,device,rng)
        chosen=torch.cat([ids['noop'][rng.integers(len(ids['noop']),size=n)],
            ids['changed'][rng.integers(len(ids['changed']),size=count-n)]])
        return {k:v[chosen].to(device) for k,v in self.splits[split]['flat'].items()}

    def sample_readout(self,split,count,device,rng):
        ids=self.indices[split];n=count//2
        if not len(ids['terminal']) or not len(ids['live']): return self.sample(split,count,device,rng)
        chosen=torch.cat([ids['terminal'][rng.integers(len(ids['terminal']),size=n)],
            ids['live'][rng.integers(len(ids['live']),size=count-n)]])
        return {k:v[chosen].to(device) for k,v in self.splits[split]['flat'].items()}

    def sample_sequence(self,split,count,steps,device,rng):
        if steps<1 or steps>self.horizon: raise ValueError('Invalid sequence window')
        if steps==1:
            batch=self.sample_prior(split,count,device,rng) if split=='train' else self.sample(split,count,device,rng)
            return {k:v[:,None] for k,v in batch.items()}
        seq=self.splits[split]['sequence'];index=self.indices[split]
        # Uniform transition starts plus long windows: preserve depth-H coverage.
        starts=index['starts'][rng.integers(len(index['starts']),size=count)].clone()
        long=(index['lengths']>=steps).nonzero().flatten();n=count//2
        if len(long):
            starts[:n,0]=long[rng.integers(len(long),size=n)]
            starts[:n,1]=torch.tensor([rng.integers(int(index['lengths'][i])-steps+1) for i in starts[:n,0]])
        times=starts[:,1,None]+torch.arange(steps);inside=times<self.horizon
        times=times.clamp_max(self.horizon-1);rows=starts[:,0,None]
        result={k:v[rows,times].clone() for k,v in seq.items()}
        result['valid']*=inside
        return {k:v.to(device) for k,v in result.items()}

    def sample_heads(self,split,count,device,rng,augment=True):
        part=self.heads[split];positive=part['dones'].nonzero().flatten();negative=(~part['dones']).nonzero().flatten()
        n=count//2
        ids=torch.cat([positive[rng.integers(len(positive),size=n)],negative[rng.integers(len(negative),size=count-n)]])
        boards=part['boards'][ids].clone();masks=part['masks'][ids].clone()
        if split=='train' and augment:
            for i in range(count):
                # Head-only rank bijections preserve equality and zeros. Never
                # apply this relabeling to merge dynamics or chance labels.
                mapping=np.arange(32,dtype=np.uint8)
                mapping[1:self.max_rank+1]=rng.permutation(np.arange(1,self.max_rank+1))
                k=int(rng.integers(4));mirror=bool(rng.integers(2))
                grid=np.rot90(mapping[boards[i].numpy()].reshape(4,4),k)
                candidate=(grid[:,::-1] if mirror else grid).copy().reshape(16)
                if canonical(candidate) not in self.forbidden:
                    boards[i]=torch.from_numpy(candidate)
                    # LEFT,UP,RIGHT,DOWN: counterclockwise rotation maps a -> a-k.
                    masks[i]=torch.roll(masks[i],-k)
                    if mirror: masks[i]=masks[i][[2,1,0,3]]
        return dict(states=boards.to(device),masks=masks.to(device),dones=part['dones'][ids].to(device),
                    weights=torch.ones(count,device=device))

    def tokenizer_boards(self,split,count,device,rng):
        boards=self.base.tokenizer_boards(split,count,device,rng)
        heads=self.sample_heads(split,count,device,rng,augment=False)
        return torch.cat([boards,heads['states']])
