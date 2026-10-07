"""All-branch supervised fitting and validation for the synthetic curriculum."""
import numpy as np
import torch
from torch.nn import functional as F

from .diagnostics import one_step_diagnostics
from .data import branch_targets
from .objectives import state_readout_loss,readout_alignment,head_coherence,changed_nll


def full_branch_loss_chunks(model,batch,chunk_size=64,latent_scale=.1,enhanced=False,metrics=None,initial_latent=None):
    """Yield independent graphs so full-support training has bounded GPU memory.

    Each source state has equal weight, and each of its supported conditional
    mappings has equal weight. The separate prior loss uses true probabilities.
    Recompute F per chunk: gradients reach F/G/heads without retaining all graphs.
    """
    world=model.world
    support=batch['chance_probs']>0
    rows,events=support.nonzero(as_tuple=True)
    targets=branch_targets(batch['afterstates'],batch['chance_probs'])
    if enhanced:
        with torch.no_grad(): z=world.encode(batch['states']) if initial_latent is None else initial_latent.detach()
        u=world.afterstate(z,batch['actions']);logp=world.chance_prior(u,z,batch['actions']).log_softmax(-1)
        prior=-(batch['chance_probs']*logp).sum(-1).mean()
        changed=changed_nll(logp,batch['chance_probs']).mean()
        if metrics is not None:
            metrics['branch_prior_ce']=float(prior.detach());metrics['branch_changed_nll']=float(changed.detach())
        yield prior+changed
    for offset in range(0,len(rows),chunk_size):
        ids,c=rows[offset:offset+chunk_size],events[offset:offset+chunk_size]
        with torch.no_grad():
            z=world.encode(batch['states'][ids]) if initial_latent is None else initial_latent[ids].detach()
            ranks=targets[ids,c]
            target_z=world.encode(ranks)
        u=world.afterstate(z,batch['actions'][ids])
        nxt=world.event_step(u,c)
        logits=world.decode(nxt)
        ce=F.cross_entropy(logits.transpose(1,2),ranks.long(),reduction='none').sum(-1)
        consistency=(nxt-target_z).square().mean((-1,-2))/(latent_scale**2)
        reward,done,legal=world.predict_outputs(u,c,nxt)
        rewards=torch.where(c==32,0,1+c%2)
        heads=F.cross_entropy(reward,rewards,reduction='none')
        heads=heads+F.binary_cross_entropy_with_logits(done,batch['branch_dones'][ids,c].float(),reduction='none')
        heads=heads+F.binary_cross_entropy_with_logits(legal,batch['branch_masks'][ids,c].float(),reduction='none').sum(-1)
        if enhanced:
            rt,rl=world.state_heads(target_z)
            encoded=state_readout_loss(rt,rl,batch['branch_dones'][ids,c],batch['branch_masks'][ids,c])
            agreement=readout_alignment(rt,rl,done,legal)
            heads=heads+encoded+.25*(agreement+head_coherence(rt,rl)+head_coherence(done,legal))
        weights=1/(len(batch['states'])*support.sum(-1)[ids])
        if metrics is not None and enhanced:
            for key,value in [('branch_encoded_head_nll',encoded),('branch_head_alignment',agreement)]:
                metrics[key]=metrics.get(key,0.)+float((value.detach()*weights).sum())
        yield ((ce+consistency+heads)*weights).sum()


@torch.no_grad()
def evaluate(model,data,stage,samples=256,batch_size=16,seed=9000000,split='validation',full_heads=True):
    was_training=model.training;model.eval()
    device=next(model.parameters()).device
    rng=np.random.default_rng(seed)
    # Fixed, separate reconstruction probe: frozen E/D see the same acceptance
    # set at every stage, independent of extra dynamics RNG draws.
    enhanced=getattr(data,'optimized',False)
    recon_rng=np.random.default_rng((39000000 if split=="validation" else 49000000)+(seed if enhanced else 0))
    total={};counts={};depth={}
    def add(key,numerator,denominator):
        total[key]=total.get(key,0.)+float(numerator)
        counts[key]=counts.get(key,0.)+float(denominator)
    try:
        for start in range(0,samples,batch_size):
            n=min(batch_size,samples-start)
            # Include rare branch outcomes and terminal roots in reconstruction.
            boards=data.tokenizer_boards(split,max(n,2),device,recon_rng)
            reconstructed=model.world.decode(model.world.encode(boards)).argmax(-1)
            add('reconstruction',(reconstructed==boards).all(-1).sum(),len(boards))
            if stage=='tokenizer': continue
            horizon={'world1':1,'world3':3,'world10':10}[stage]
            batch=data.sample_sequence(split,n,horizon,device,rng)
            z=model.world.encode(batch['states'][:,0])
            rollout_correct=torch.ones(n,dtype=torch.bool,device=device)
            for t in range(horizon):
                valid=batch['valid'][:,t].bool(); count=int(valid.sum())
                if not count: continue
                u=model.world.afterstate(z,batch['actions'][:,t])
                nxt=model.world.event_step(u,batch['events'][:,t])
                after_ok=(model.world.decode(u).argmax(-1)==batch['afterstates'][:,t]).all(-1)
                next_ok=(model.world.decode(nxt).argmax(-1)==batch['next_states'][:,t]).all(-1)
                # Preserve every earlier failure across the whole rollout.
                rollout_correct &= after_ok & next_ok
                add('afterstate',(after_ok & valid).sum(),count)
                add('next_state',(next_ok & valid).sum(),count)
                for key,values in [('afterstate',after_ok),('next_state',next_ok),('prefix',rollout_correct)]:
                    name=f'step_{t+1}_{key}';add(name,(values & valid).sum(),count)
                depth[str(t+1)]=depth.get(str(t+1),0)+count
                p=model.world.event_probabilities(z,u,batch['actions'][:,t])
                q=batch['chance_probs'][:,t]
                add('prior_tv',(.5*(p-q).abs().sum(-1))[valid].sum(),count)
                add('invalid_event_mass',(p*(q==0)).sum(-1)[valid].sum(),count)
                reward,done,legal=model.world.predict_outputs(u,batch['events'][:,t],nxt)
                add('reward',((reward.argmax(-1)==(batch['rewards'][:,t]/2).long()) & valid).sum(),count)
                add('generated_legal',(((legal>=0)==batch['next_masks'][:,t]).all(-1)&valid).sum(),count)
                if enhanced:
                    rt,rl=model.world.state_heads(model.world.encode(batch['next_states'][:,t]))
                    add('encoded_next_legal',(((rl>=0)==batch['next_masks'][:,t]).all(-1)&valid).sum(),count)
                    add('generated_next_legal',(((legal>=0)==batch['next_masks'][:,t]).all(-1)&valid).sum(),count)
                    agreement=((rl>=0)==(legal>=0)).all(-1)&((rt>=0)==(done>=0))
                    add('head_agreement',(agreement&valid).sum(),count)
                    pos=batch['dones'][:,t]&valid;neg=~batch['dones'][:,t]&valid
                    for metric_prefix,d in [('encoded_next',rt),('generated_next',done)]:
                        add(metric_prefix+'_terminal_recall',(d[pos]>=0).sum(),pos.sum())
                        add(metric_prefix+'_terminal_false_positive',(d[neg]>=0).sum(),neg.sum())
                    for metric_prefix,select in [('noop',q[:,32]>0),('changed',q[:,32]==0)]:
                        select &= valid
                        add(metric_prefix+'_invalid_mass',(p*(q==0)).sum(-1)[select].sum(),select.sum())
                z=nxt
            if not enhanced or not full_heads:
                heads=data.sample_heads(split,max(n,2),device,rng,**({'augment':False} if enhanced else {}))
                done,legal=model.world.state_heads(model.world.encode(heads['states']))
                positive=heads['dones'];negative=~positive
                add('terminal_recall',(done[positive]>=0).sum(),positive.sum())
                add('terminal_false_positive',(done[negative]>=0).sum(),negative.sum())
                add('legal',((legal>=0)==heads['masks']).all(-1).sum(),len(done))
            point=data.sample(split,n,device,rng)
            # Enumerates every model event, aggregates equal decoded boards, and
            # compares to independent Board rules (including unsupported mass).
            diag=one_step_diagnostics(model,point)
            for key in ('decoded_invalid_mass','decoded_branch_tv'):
                add(key,diag[key]*n,n)
            u=model.world.afterstate(model.world.encode(point['states']),point['actions'])
            targets=branch_targets(point['afterstates'],point['chance_probs'])
            support=point['chance_probs']>0
            ids,c=support.nonzero(as_tuple=True)
            correct=[]
            for offset in range(0,len(ids),64):
                ii,cc=ids[offset:offset+64],c[offset:offset+64]
                pred=model.world.decode(model.world.event_step(u[ii],cc)).argmax(-1)
                correct.append((pred==targets[ii,cc]).all(-1))
            if correct: add('all_branch_accuracy',torch.cat(correct).sum(),len(ids))
        if enhanced and stage!='tokenizer':
            # Dedicated conditional probes prevent a small fixed validation
            # draw from missing no-op or terminal examples indefinitely.
            for start in range(0,max(samples,128),max(batch_size,4)):
                n=min(max(batch_size,4),max(samples,128)-start)
                probe=data.sample_prior(split,n,device,rng)
                z=model.world.encode(probe['states']);u=model.world.afterstate(z,probe['actions'])
                p=model.world.event_probabilities(z,u,probe['actions']);q=probe['chance_probs']
                for metric_prefix,select in [('noop',q[:,32]>0),('changed',q[:,32]==0)]:
                    add(metric_prefix+'_invalid_mass',(p*(q==0)).sum(-1)[select].sum(),select.sum())
                probe=data.sample_readout(split,n,device,rng)
                z=model.world.encode(probe['states']);u=model.world.afterstate(z,probe['actions'])
                nxt=model.world.event_step(u,probe['events'])
                done,legal=model.world.state_heads(nxt)
                rt,rl=model.world.state_heads(model.world.encode(probe['next_states']))
                positive=probe['dones'];negative=~positive
                add('encoded_next_legal',((rl>=0)==probe['next_masks']).all(-1).sum(),n)
                add('generated_next_legal',((legal>=0)==probe['next_masks']).all(-1).sum(),n)
                add('head_agreement',(((rl>=0)==(legal>=0)).all(-1)&((rt>=0)==(done>=0))).sum(),n)
                for metric_prefix,d in [('encoded_next',rt),('generated_next',done)]:
                    add(metric_prefix+'_terminal_recall',(d[positive]>=0).sum(),positive.sum())
                    add(metric_prefix+'_terminal_false_positive',(d[negative]>=0).sum(),negative.sum())
        if enhanced and full_heads:
            part=data.heads[split]
            for start in range(0,len(part['boards']),max(batch_size,64)):
                boards=part['boards'][start:start+max(batch_size,64)].to(device)
                labels=part['dones'][start:start+len(boards)].to(device)
                masks=part['masks'][start:start+len(boards)].to(device)
                encoded=model.world.encode(boards)
                add('reconstruction',(model.world.decode(encoded).argmax(-1)==boards).all(-1).sum(),len(boards))
                if stage=='tokenizer': continue
                done,legal=model.world.state_heads(encoded)
                add('terminal_recall',(done[labels]>=0).sum(),labels.sum())
                add('terminal_false_positive',(done[~labels]>=0).sum(),(~labels).sum())
                add('legal',((legal>=0)==masks).all(-1).sum(),len(done))
                add('root_head_agreement',((done>=0)==~(legal>=0).any(-1)).sum(),len(done))
        metrics={k:total[k]/counts[k] for k in total if counts[k]}
        metrics['depth_samples']=depth
        return metrics
    finally:
        model.train(was_training)


def gates(stage,metrics,enhanced=False):
    criteria={'reconstruction':('min',.995)}
    if stage!='tokenizer':
        horizon={'world1':1,'world3':3,'world10':10}[stage]
        accuracy=.99 if horizon==1 else (.97 if horizon==3 else .95)
        criteria.update(afterstate=('min',accuracy),next_state=('min',accuracy),
            all_branch_accuracy=('min',.99),prior_tv=('max',.03),invalid_event_mass=('max',.01),
            decoded_invalid_mass=('max',.01),decoded_branch_tv=('max',.05),
            reward=('min',.99),legal=('min',.99),generated_legal=('min',.99),
            terminal_recall=('min',.95),terminal_false_positive=('max',.01))
        criteria[f'step_{horizon}_prefix']=('min',accuracy)
        if enhanced:
            criteria.update(encoded_next_legal=('min',.99),generated_next_legal=('min',.99),head_agreement=('min',.99),
                encoded_next_terminal_recall=('min',.95),encoded_next_terminal_false_positive=('max',.01),
                generated_next_terminal_recall=('min',.95),generated_next_terminal_false_positive=('max',.01),
                root_head_agreement=('min',.99),noop_invalid_mass=('max',.01),changed_invalid_mass=('max',.01))
    checks={k:dict(value=metrics.get(k),direction=direction,threshold=threshold,
                  passed=bool(k in metrics and np.isfinite(metrics[k]) and
                              (metrics[k]>=threshold if direction=='min' else metrics[k]<=threshold)))
            for k,(direction,threshold) in criteria.items()}
    return dict(passed=all(row['passed'] for row in checks.values()),checks=checks,
                failures=[k for k,v in checks.items() if not v['passed']])
