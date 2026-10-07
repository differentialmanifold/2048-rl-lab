"""Differentiable supervised, multi-step neural dynamics fitting (no rules)."""
import torch
from torch.nn import functional as F


def state_readout_loss(terminal,legal,dones,masks):
    return (F.binary_cross_entropy_with_logits(terminal,dones.float(),reduction='none')+
            F.binary_cross_entropy_with_logits(legal,masks.float(),reduction='none').sum(-1))


def head_coherence(terminal,legal):
    # Training regularizer only; inference remains two independent predictions.
    return (terminal.sigmoid()-(1-legal.sigmoid()).prod(-1)).square()


def readout_alignment(real_terminal,real_legal,terminal,legal):
    return ((real_terminal.sigmoid()-terminal.sigmoid()).square()+
            (real_legal.sigmoid()-legal.sigmoid()).square().sum(-1))


def changed_nll(logp,q):
    return -(q[:,32]*logp[:,32]+(1-q[:,32])*torch.logsumexp(logp[:,:32],-1))


def supervised_world_loss(model, batch, heads=None, latent_scale=.1, consistency_weight=1.,enhanced=False,
                          initial_latent=None, head_margin=0., invalid_weight=0.):
    world = model.world
    if batch['states'].ndim == 2:
        batch = {k:v[:,None] for k,v in batch.items()}
        batch['valid'] = torch.ones_like(batch['actions'],dtype=torch.float32)
    b, steps = batch['actions'].shape
    valid = batch['valid'].float()
    def average(value):
        return (value * valid).sum()/valid.sum().clamp_min(1)
    with torch.no_grad():
        z = world.encode(batch['states'][:,0]) if initial_latent is None else initial_latent.detach()
        if z.shape[:2] != (b,16):
            raise ValueError('Initial recurrent latent must match the sequence batch')
        after_target = world.encode(batch['afterstates'].flatten(0,1)).reshape(b,steps,16,-1)
        next_target = world.encode(batch['next_states'].flatten(0,1)).reshape(b,steps,16,-1)
    terms = {k:[] for k in ('afterstate_nll','next_state_nll','event_prior_nll','reward_nll',
        'terminal_nll','legal_nll','latent_consistency','afterstate_board_accuracy',
        'conditional_next_board_accuracy','generated_legal_mask_accuracy','reward_accuracy','reward_mean_mae',
        'prior_entropy','prior_spawn4_mass')}
    if enhanced:
        terms.update({k:[] for k in ('encoded_head_nll','head_alignment_loss','head_coherence_loss',
            'changed_nll','prior_kl','invalid_event_mass')})
    if initial_latent is not None:
        terms.update({k:[] for k in ('recurrent_root_head_nll','recurrent_head_margin','root_readout_alignment')})
    for t in range(steps):
        if initial_latent is not None:
            # Supervise the *current recurrent* state, including false-positive
            # actions. Reconstructing the board alone cannot verify this head.
            dt,dl=world.state_heads(z)
            with torch.no_grad(): encoded=world.encode(batch['states'][:,t])
            et,el=world.state_heads(encoded)
            masks=batch['masks'][:,t];dones=~masks.any(-1)
            terms['recurrent_root_head_nll'].append(state_readout_loss(dt,dl,dones,masks)+
                                                   state_readout_loss(et,el,dones,masks))
            labels=2*masks.float()-1
            margin=(F.relu(head_margin-labels*dl).square()+F.relu(head_margin-labels*el).square()).sum(-1)
            terms['recurrent_head_margin'].append(margin)
            terms['root_readout_alignment'].append(readout_alignment(et,el,dt,dl))
        actions, events = batch['actions'][:,t], batch['events'][:,t]
        u = world.afterstate(z, actions)
        prior_logits = world.chance_prior(u, z, actions)
        logp = prior_logits.log_softmax(-1)
        terms['prior_entropy'].append(-(logp.exp()*logp).sum(-1))
        terms['prior_spawn4_mass'].append(logp.exp()[:,1:32:2].sum(-1))
        # Only the observed event label is forced. The next input state remains
        # the model prediction, so later losses train recursive latent dynamics.
        predicted = world.event_step(u, events)
        after_logits, next_logits = world.decode(u), world.decode(predicted)
        def board_loss(logits, target):
            return F.cross_entropy(logits.transpose(1,2), target.long(), reduction='none').sum(-1)
        terms['afterstate_nll'].append(board_loss(after_logits,batch['afterstates'][:,t]))
        terms['next_state_nll'].append(board_loss(next_logits,batch['next_states'][:,t]))
        if 'chance_probs' in batch:
            # Exact teacher distribution, including zero probability on illegal
            # events. Never replace this with equal weights on enumerated branches.
            terms['event_prior_nll'].append(-(batch['chance_probs'][:,t]*logp).sum(-1))
        else:
            terms['event_prior_nll'].append(F.cross_entropy(prior_logits,events,reduction='none'))
        reward, terminal, legal = world.predict_outputs(u,events,predicted)
        if enhanced:
            rt,rl=world.state_heads(next_target[:,t])
            terms['encoded_head_nll'].append(state_readout_loss(rt,rl,batch['dones'][:,t],batch['next_masks'][:,t]))
            terms['head_alignment_loss'].append(readout_alignment(rt,rl,terminal,legal))
            terms['head_coherence_loss'].append(head_coherence(rt,rl)+head_coherence(terminal,legal))
            q=batch['chance_probs'][:,t]
            terms['changed_nll'].append(changed_nll(logp,q))
            terms['prior_kl'].append((q*(q.clamp_min(1e-30).log()-logp)).sum(-1))
            terms['invalid_event_mass'].append((logp.exp()*(q==0)).sum(-1))
        terms['reward_nll'].append(F.cross_entropy(reward,(batch['rewards'][:,t]/2).long(),reduction='none'))
        terms['terminal_nll'].append(F.binary_cross_entropy_with_logits(terminal,batch['dones'][:,t].float(),reduction='none'))
        terms['legal_nll'].append(F.binary_cross_entropy_with_logits(legal,batch['next_masks'][:,t].float(),reduction='none').sum(-1))
        error = ((u-after_target[:,t]).square().mean((-1,-2))
                 + (predicted-next_target[:,t]).square().mean((-1,-2)))/(2*latent_scale**2)
        terms['latent_consistency'].append(error)
        terms['afterstate_board_accuracy'].append((after_logits.argmax(-1)==batch['afterstates'][:,t]).all(-1).float())
        terms['conditional_next_board_accuracy'].append((next_logits.argmax(-1)==batch['next_states'][:,t]).all(-1).float())
        terms['generated_legal_mask_accuracy'].append(((legal>=0)==batch['next_masks'][:,t]).all(-1).float())
        terms['reward_accuracy'].append((reward.argmax(-1)==(batch['rewards'][:,t]/2).long()).float())
        expected_reward = (reward.softmax(-1)*world.reward_values).sum(-1)
        terms['reward_mean_mae'].append((expected_reward-batch['rewards'][:,t]).abs())
        z = predicted
    reduced = {k:average(torch.stack(v,1)) for k,v in terms.items()}
    likelihood = sum(reduced[k] for k in ('afterstate_nll','next_state_nll','event_prior_nll',
                                        'reward_nll','terminal_nll','legal_nll'))
    loss = likelihood + consistency_weight*reduced['latent_consistency']
    if enhanced:
        loss=loss+reduced['encoded_head_nll']+reduced['changed_nll']+.25*(
            reduced['head_alignment_loss']+reduced['head_coherence_loss'])
        loss=loss+invalid_weight*reduced['invalid_event_mass']
    if initial_latent is not None:
        loss=loss+reduced['recurrent_root_head_nll']+reduced['root_readout_alignment']
        if head_margin>0: loss=loss+.25*reduced['recurrent_head_margin']
    metrics = {k:v.detach() for k,v in reduced.items()}
    metrics['supervised_joint_nll'] = likelihood.detach()
    metrics['sequence_transitions'] = valid.sum()
    if heads is not None:
        with torch.no_grad():
            encoded = world.encode(heads['states'])
        terminal, legal = world.state_heads(encoded)
        labels, weights = heads['dones'].float(), heads['weights']
        terminal_loss = F.binary_cross_entropy_with_logits(terminal,labels,reduction='none')
        legal_loss = F.binary_cross_entropy_with_logits(legal,heads['masks'].float(),reduction='none').sum(-1)
        head_loss = ((terminal_loss+legal_loss)*weights).mean()
        if enhanced: head_loss=head_loss+.25*head_coherence(terminal,legal).mean()
        loss = loss + head_loss
        positive = heads['dones']
        metrics.update(head_loss=head_loss.detach(),
            terminal_recall=((terminal[positive]>=0).float().mean()).detach(),
            terminal_false_positive_rate=((terminal[~positive]>=0).float().mean()).detach(),
            terminal_brier=(((terminal.sigmoid()-labels).square()*weights).mean()).detach(),
            head_legal_mask_accuracy=((((legal>=0)==heads['masks']).all(-1).float()*weights).mean()).detach())
    return loss, metrics
