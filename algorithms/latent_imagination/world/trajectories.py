"""Fit world dynamics and readouts on long policy trajectories.

The source run and policy are snapshots. Rules and decoding label offline data
and audit predictions only; the repaired world's inference is still neural.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from common.checkpoints import read_checkpoint, model_from_checkpoint, save_checkpoint, restore_checkpoint
from common.parallel import GamePool
from common.training import setup
from common.models import model_from_config
from ..runtime import file_digest
from .dynamics import world_fingerprint
from .objectives import supervised_world_loss
from .trajectory_data import prepare_probes, ProbeData, policy_trace
from .data import SyntheticData, atomic_json
from .boundaries import CurriculumData
from .branches import evaluate, gates, full_branch_loss_chunks
from .verification import verify_latent_game, save_game_report

DEFAULTS=dict(seed=0,device='mps',workers=8,episodes=32,probe_stride=8,max_steps=10000,
    updates=1000,max_updates=0,eval_every=200,plot_every=25,refresh_every=1000,
    batch_size=16,horizon=10,lr=1e-4,dynamics_lr=1e-5,consistency_weight=4.,head_margin=2.,
    invalid_weight=20.,rehearsal_weight=1.,validation_samples=1024,audit_games=8,
    regression_seeds='1114000002',world_run='',policy_checkpoint='')
RUNTIME={'device','workers','max_updates','updates'}


def stage_directory(root):
    # Existing runs preserve their on-disk paths; fresh worlds use a stage name.
    return root / ('repair' if (root / 'repair').exists() else 'trajectories')


def configuration(args,root):
    old=json.loads((root/'run.json').read_text()) if (root/'run.json').exists() else None
    if old and old.get('world_repair_version')!=2:raise ValueError('Use a new trajectory stage directory for the shared-event legal head')
    cfg={**DEFAULTS,**(old['config'] if old else {})}
    for k in DEFAULTS:
        value=getattr(args,k)
        if value is None:continue
        if k in ('world_run','policy_checkpoint'):value=str(Path(value).resolve())
        if old and k not in RUNTIME and value!=cfg[k]:raise ValueError(f'Trajectory resume must preserve {k}')
        if old and k=='updates' and value<cfg[k]:raise ValueError('Update target cannot decrease')
        cfg[k]=value
    for k,v in cfg.items():
        if k in ('device','regression_seeds','world_run','policy_checkpoint'):continue
        if not math.isfinite(v) or v< (0 if k in ('seed','max_updates') else 1e-12):
            raise ValueError(f'Invalid repair option: {k}')
    if cfg['episodes']<4 or cfg['audit_games']<2 or cfg['horizon']>10:
        raise ValueError('Need episodes>=4, audit-games>=2 and horizon<=10')
    if not cfg['world_run'] or not cfg['policy_checkpoint']:raise ValueError('Require world-run and policy-checkpoint')
    cfg['regression_seeds']=','.join(str(int(s)) for s in cfg['regression_seeds'].split(',') if s.strip())
    if not old:
        from .source import verified_source
        source,provenance=verified_source(cfg['world_run'])
        # Open a moving last.pt once: saving its bytes makes provenance stable.
        (root/'teacher.pt').write_bytes(Path(cfg['policy_checkpoint']).read_bytes())
        teacher=read_checkpoint(root/'teacher.pt')
        if teacher['algorithm']!='latent_afterstate_ppo' or teacher.get('verified_world_sha256')!=provenance['world_sha256']:
            raise ValueError('Policy must use latent imagination using this verified world')
        if teacher['model_config']['world_config']!=source['model_config']:
            raise ValueError('Teacher and world source must use identical world semantics')
        torch.save(source,root/'source.pt')
        atomic_json(root/'source.json',dict(world=provenance,teacher_iteration=teacher['iteration'],
            teacher_sha256=file_digest(root/'teacher.pt'),source_sha256=file_digest(root/'source.pt')))
        for name in ('data','head_data'):
            target=Path(cfg['world_run'])/name
            if (root/name).is_symlink() and (root/name).resolve()==target.resolve():continue
            (root/name).symlink_to(target,target_is_directory=True)
    source=json.loads((root/'source.json').read_text())
    if (file_digest(root/'teacher.pt')!=source['teacher_sha256'] or
        file_digest(root/'source.pt')!=source['source_sha256']):raise ValueError('World input snapshot changed')
    atomic_json(root/'run.json',dict(world_repair_version=2,config=cfg))
    return cfg,source


def prepare(root,cfg,data,round_index=0,world_path=None):
    directory=stage_directory(root)/f'probes_{round_index:04d}'
    return prepare_probes(directory,root/'teacher.pt',world_path or root/'source.pt',data,
        cfg['episodes'],cfg['seed'],cfg['workers'],cfg['max_steps'],cfg['probe_stride'],
        [int(s) for s in cfg['regression_seeds'].split(',') if s],round_index)


def optimizer_for(model,cfg):
    model.set_phase('world')
    heads=[];dynamics=[]
    for name,p in model.world.named_parameters():
        if not p.requires_grad:continue
        (dynamics if name.startswith(('action_transition.','event_transition.','action_embedding.','event_embedding.')) else heads).append(p)
    return torch.optim.Adam([dict(params=heads,lr=cfg['lr']),dict(params=dynamics,lr=cfg['dynamics_lr'])])


@torch.no_grad()
def probe_metrics(model,probes,count=1024):
    device=next(model.parameters()).device;rows=probes.splits['validation']
    # Fixed, evenly distributed roots; the validation corpus is never sampled by
    # training. Evaluate stale long latents AND the re-encoded counterparts.
    ids=np.linspace(0,len(rows)-1,min(count,len(rows)),dtype=int)
    correct=encoded_correct=agreement=false_positive=negative=0
    from .verification import reference
    for start in range(0,len(ids),64):
        group=[rows[i] for i in ids[start:start+64]]
        z=torch.stack([r['latent'] for r in group]).to(device)
        boards=torch.stack([r['board'] for r in group]).to(device)
        masks=torch.tensor([reference(r['board'].numpy())['mask'] for r in group],device=device)
        dt,dl=model.world.state_heads(z);et,el=model.world.state_heads(model.world.encode(boards))
        correct+=int(((dl>=0)==masks).all(-1).sum());encoded_correct+=int(((el>=0)==masks).all(-1).sum())
        agreement+=int(((dl>=0)==(el>=0)).all(-1).sum())
        false_positive+=int(((dl>=0)&~masks).sum());negative+=int((~masks).sum())
    return dict(samples=len(ids),recurrent_legal=correct/len(ids),encoded_legal=encoded_correct/len(ids),
        readout_agreement=agreement/len(ids),illegal_false_positive=false_positive/max(1,negative))


def audit_job(job):
    teacher_path,world_path,seed,max_steps,random_policy,output=job
    setup(seed,'cpu');model=model_from_checkpoint(read_checkpoint(teacher_path)).eval()
    model.world=model_from_checkpoint(read_checkpoint(world_path)).world
    model.world.requires_grad_(False);model.world.eval()
    trace=policy_trace(model,seed,max_steps,random_policy)
    report=verify_latent_game(model.world,trace,.01,.05)
    if output:save_game_report(report,output)
    result={k:v for k,v in report.items() if k not in ('trajectory','errors')}
    result.update(random_policy=random_policy,zero_rewards=sum(float(s['reward'])==0 for s in trace['steps']))
    return result


def long_audit(root,cfg,checkpoint,iteration):
    out=stage_directory(root)/f'audit_{iteration:08d}';out.mkdir(exist_ok=True)
    specs=[(9200000+cfg['seed']+i,False,'policy') for i in range(cfg['audit_games'])]
    specs += [(9300000+cfg['seed']+i,True,'random') for i in range(2)]
    specs += [(int(s),False,'regression') for s in cfg['regression_seeds'].split(',') if s]
    jobs=[(str(root/'teacher.pt'),str(checkpoint),seed,cfg['max_steps'],random,
           str(out/f'{role}_{seed}.json') if role=='regression' or i==0 else None)
          for i,(seed,random,role) in enumerate(specs)]
    with GamePool(cfg['workers']) as pool:games=pool.map(audit_job,jobs)
    for game,(_,_,role) in zip(games,specs):game['role']=role
    atomic_json(out/'summary.json',games)
    return games


def trajectory_gates(base,probes,games):
    checks=dict(base=base['passed'],recurrent_legal=probes['recurrent_legal']>=.999,
        encoded_legal=probes['encoded_legal']>=.999,readout_agreement=probes['readout_agreement']>=.999,
        illegal_false_positive=probes['illegal_false_positive']<=.0001,
        policy_games=bool([g for g in games if g['role']=='policy']) and
            all(g['passed'] for g in games if g['role']=='policy'),
        random_games=bool([g for g in games if g['role']=='random']) and
            all(g['passed'] for g in games if g['role']=='random'),
        regressions=all(g['passed'] for g in games if g['role']=='regression'))
    return dict(passed=all(checks.values()),checks=checks,failures=[k for k,v in checks.items() if not v])


def plots(root,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    train=[r for r in history if 'loss' in r];vals=[r for r in history if 'validation' in r]
    if train:
        axes[0].plot([r['iteration'] for r in train],[r['loss'] for r in train]);axes[0].set_yscale('log')
    if vals:
        x=[r['iteration'] for r in vals]
        for key in ('recurrent_legal','encoded_legal','readout_agreement'):
            axes[1].plot(x,[r['validation']['probes'][key] for r in vals],label=key)
        axes[1].axhline(.999,ls='--',color='grey');axes[1].legend(fontsize=7)
        axes[2].plot(x,[sum(g['passed'] for g in r['validation']['games'])/len(r['validation']['games']) for r in vals])
    for ax,title in zip(axes,['Trajectory + rehearsal loss','Held-out long-latent heads','Whole-game strict pass fraction']):
        ax.set(title=title,xlabel='World trajectory update');ax.grid(alpha=.2)
    axes[1].set_ylim(0,1.01);axes[2].set_ylim(0,1.01)
    fig.savefig(stage_directory(root)/'training.png',dpi=140);plt.close(fig)


def run(root,cfg,data,probes,validate_only=False):
    device=setup(cfg['seed'],cfg['device']);directory=stage_directory(root);last=directory/'last.pt'
    saved=read_checkpoint(last if last.exists() else root/'source.pt')
    resumed=last.exists()
    if resumed:
        if saved.get('world_repair_version')!=2:raise ValueError('Old trajectory objective; use a new run directory')
        model=model_from_checkpoint(saved,device)
    else:
        config={**saved['model_config'],'state_head_version':3}
        model=model_from_config(config).to(device)
        missing,unexpected=model.load_state_dict(saved['model'],strict=False)
        if unexpected or any(not k.startswith('world.state_refiner.') for k in missing):
            raise ValueError('Unsupported trajectory initializer')
    optimizer=optimizer_for(model,cfg)
    iteration=saved['iteration'] if resumed else 0;streak=saved.get('repair_streak',0) if resumed else 0
    active_round=saved.get('repair_probe_round',0) if resumed else 0
    last_validation=saved.get('repair_last_validation',-1) if resumed else -1
    if resumed:restore_checkpoint(saved,model,optimizer,restore_rng=True)
    if (not validate_only and resumed and iteration>=cfg['updates'] and streak>=2
            and (root/'world10/passed.pt').exists() and file_digest(last)==file_digest(root/'world10/passed.pt')):
        require_trajectory_report(root,saved)
        return True
    if active_round:
        probes=ProbeData(directory/f'probes_{active_round:04d}',data)
    for index in range(active_round+1):
        previous=ProbeData(directory/f'probes_{index:04d}',data)
        data.forbidden.update(previous.forbidden)
    probes.forbidden.update(data.forbidden)
    history=[]
    logs=directory/'metrics.jsonl'
    if logs.exists():
        history=[json.loads(s) for s in logs.read_text().splitlines() if s.strip()]
        history=[r for r in history if r['iteration']<=iteration]
        logs.write_text(''.join(json.dumps(r)+'\n' for r in history))
    def persist():
        save_checkpoint(last,model,optimizer,iteration,'latent_dreamer',
            {**cfg,'architecture':'latent_transformer','phase':'world'},0.,dict(world_repair_version=2,
            dataset_sha256=data.digest,repair_probe_sha256=probes.digest,repair_probe_round=active_round,
            repair_streak=streak,repair_last_validation=last_validation,
            repair_teacher_sha256=file_digest(root/'teacher.pt')))
    def record(row):
        history.append(row)
        with logs.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        compact={k:v for k,v in row.items() if k!='validation'}
        if 'validation' in row:compact.update(passed=row['validation']['passed'],failures=row['validation']['failures'])
        print(json.dumps(compact,allow_nan=False),flush=True)
    while True:
        should_check=validate_only or (iteration!=last_validation and
            (iteration%cfg['eval_every']==0 or (cfg['max_updates'] and iteration>=cfg['max_updates'])))
        if should_check:
            # A stable temporary checkpoint is read by CPU audit workers.
            persist()
            metrics=evaluate(model,data,'world10',cfg['validation_samples'],cfg['batch_size'],cfg['seed']+9000000)
            base=gates('world10',metrics,True);pm=probe_metrics(model,probes,cfg['validation_samples'])
            games=long_audit(root,cfg,last,iteration)
            report=trajectory_gates(base,pm,games)
            if iteration!=last_validation:streak=streak+1 if report['passed'] else 0
            last_validation=iteration;persist()
            fingerprint=world_fingerprint(model.world);digest=file_digest(last)
            report.update(iteration=iteration,consecutive_passes=streak,base=base,metrics=metrics,probes=pm,games=games,
                data_sha256=data.digest,world_sha256=fingerprint,checkpoint_sha256=digest,
                probe_sha256=probes.digest,probe_round=active_round,teacher_sha256=file_digest(root/'teacher.pt'))
            atomic_json(directory/'validation.json',report);record(dict(iteration=iteration,validation=report));plots(root,history)
            passed=report['passed'] and streak>=2 and iteration>=cfg['updates']
            atomic_json(root/'progress.json',dict(status='passed' if passed else 'training',iteration=iteration,
                failures=report['failures'],consecutive_passes=streak,policy_training_allowed=passed))
            if passed:
                (root/'world10').mkdir(exist_ok=True);(root/'audit').mkdir(exist_ok=True)
                shutil.copyfile(last,root/'world10/passed.pt')
                atomic_json(root/'world10/validation.json',dict(**base,iteration=iteration,consecutive_passes=streak,
                    metrics=metrics,data_sha256=data.digest,world_sha256=fingerprint,checkpoint_sha256=digest))
                atomic_json(root/'audit/validation.json',dict(passed=True,split='validation',games=games,
                    data_sha256=data.digest,world_sha256=fingerprint,checkpoint_sha256=digest))
                return True
            if validate_only:return False
        if cfg['max_updates'] and iteration>=cfg['max_updates']:return False
        if iteration and iteration%cfg['refresh_every']==0 and active_round<iteration//cfg['refresh_every']:
            # Only fresh TRAIN seed namespaces are fitted. Freeze previous held-
            # out roots out of all subsequent supervision, including rehearsal.
            data.forbidden.update(probes.forbidden)
            persist()
            active_round=iteration//cfg['refresh_every'];snapshot=directory/f'world_round_{active_round:04d}.pt'
            if not snapshot.exists():shutil.copyfile(last,snapshot)
            probes=prepare(root,cfg,data,active_round,snapshot);persist()
        rng=np.random.default_rng(cfg['seed']+iteration*7919)
        batch,z=probes.sample('train',cfg['batch_size'],cfg['horizon'],device,rng)
        optimizer.zero_grad(set_to_none=True)
        loss,details=supervised_world_loss(model,batch,enhanced=True,initial_latent=z,
            consistency_weight=cfg['consistency_weight'],head_margin=cfg['head_margin'],invalid_weight=cfg['invalid_weight'])
        value=float(loss.detach());loss.backward()
        rehearsal=data.sample_sequence('train',cfg['batch_size'],10,device,rng)
        replay_loss,_=supervised_world_loss(model,rehearsal,data.sample_heads('train',max(2,cfg['batch_size']),device,rng),enhanced=True)
        value+=float(replay_loss.detach())*cfg['rehearsal_weight'];(replay_loss*cfg['rehearsal_weight']).backward()
        branch_value=0.;repair_branch_value=0.
        for branch in full_branch_loss_chunks(model,data.sample_prior('train',2,device,rng),32,enhanced=True):
            branch_value+=float(branch.detach());(branch*cfg['rehearsal_weight']).backward()
        branch_batch,branch_z=probes.sample_branches(2,device,rng)
        for branch in full_branch_loss_chunks(model,branch_batch,32,enhanced=True,initial_latent=branch_z):
            repair_branch_value+=float(branch.detach());branch.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],100.)
        if not math.isfinite(value+branch_value+repair_branch_value) or not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite trajectory gradient')
        optimizer.step();iteration+=1
        if iteration%cfg['plot_every']==0 or iteration%cfg['eval_every']==0 or (cfg['max_updates'] and iteration>=cfg['max_updates']):
            persist();record(dict(iteration=iteration,loss=value+branch_value*cfg['rehearsal_weight']+repair_branch_value,
                grad_norm=float(norm),probe_round=active_round,**{k:float(v) for k,v in details.items()}));plots(root,history)


def require_trajectory_report(root,source):
    """Acceptance report for the long-trajectory world stage, called by the policy trainer."""
    if source.get('world_repair_version') not in (1,2):return None
    path=stage_directory(root)/'validation.json'
    if not path.exists():raise ValueError('Missing long-policy trajectory validation')
    report=json.loads(path.read_text())
    if source.get('world_repair_version')==2 and source['model_config'].get('state_head_version')!=3:
        raise ValueError('Trajectory-trained world requires shared-event legal semantics')
    manifest=stage_directory(root)/f'probes_{source["repair_probe_round"]:04d}'/'manifest.json'
    manifest_data=json.loads(manifest.read_text())
    if (not report.get('passed') or report.get('consecutive_passes',0)<2
        or report.get('checkpoint_sha256')!=file_digest(root/'world10/passed.pt')
        or report.get('world_sha256')!=world_fingerprint(model_from_checkpoint(source).world)
        or report.get('probe_sha256')!=source['repair_probe_sha256']
        or file_digest(manifest)!=source['repair_probe_sha256']
        or file_digest(manifest.parent/'probes.pt')!=manifest_data['sha256']
        or report.get('teacher_sha256')!=source['repair_teacher_sha256']
        or file_digest(root/'teacher.pt')!=source['repair_teacher_sha256']):
        raise ValueError('Trajectory-trained world has not passed its bound long-policy validation')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',default='run',choices=('run','prepare','validate','status'))
    p.add_argument('--run-dir',required=True)
    for key,value in DEFAULTS.items():p.add_argument('--'+key.replace('_','-'),type=type(value))
    args=p.parse_args(argv);root=Path(args.run_dir).resolve()
    if args.command=='status':print((root/'progress.json').read_text());return True
    from ..runtime import lock_run
    with lock_run(root):
        cfg,_=configuration(args,root)
        data=CurriculumData(SyntheticData(root/'data'),root/'head_data')
        (stage_directory(root)).mkdir(exist_ok=True)
        probes=prepare(root,cfg,data)
        if args.command=='prepare':return True
        return run(root,cfg,data,probes,args.command=='validate')


if __name__=='__main__':
    if main() is False:raise SystemExit(2)
