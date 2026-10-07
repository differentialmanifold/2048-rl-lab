"""Base world fitting: tokenizer, one-step and multi-step dynamics."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from common.checkpoints import read_checkpoint,model_from_checkpoint,save_checkpoint,restore_checkpoint
from common.training import setup
from common.parallel import GamePool
from ..runtime import file_digest, lock_run
from .model import WorldModel
from .objectives import supervised_world_loss
from .dynamics import world_fingerprint
from .data import generate,SyntheticData,atomic_json,transition,KINDS
from .branches import full_branch_loss_chunks,evaluate,gates
from .plots import coverage_plot,progress_plot,training_plots,reconstruction_plot
from .boundaries import CurriculumData,prepare_boundaries
from .runtime import (DEFAULTS as CURRICULUM_DEFAULTS,initializer,load_initial_model,
    copy_base_data,optimizer_for,set_repair_phase,gate_margin,plateau_state)

STAGES=('tokenizer','world1','world3','world10')
DEFAULTS=dict(seed=0,device='auto',workers=8,families=6000,horizon=10,max_rank=12,
    width=128,latent_dim=16,layers=2,heads=4,batch_size=32,branch_batch_size=8,branch_chunk_size=64,
    lr=3e-4,eval_every=200,plot_every=50,validation_samples=1024,
    audit_games=5,audit_max_steps=10000,max_updates_per_stage=0,**CURRICULUM_DEFAULTS)
RUNTIME={'device','workers','max_updates_per_stage','plateau_patience','stop_on_plateau'}


def settings(args):
    path=Path(args.run_dir)/'run.json'
    old=json.loads(path.read_text()) if path.exists() else None
    config={**DEFAULTS,**({k:v for k,v in old['config'].items() if k in DEFAULTS} if old else {})}
    if old and old.get('pipeline_version')!=3:
        raise ValueError('Legacy pipeline objective changed: use a NEW --run-dir with --init-from WORLD_CHECKPOINT and --data-from OLD_RUN/data')
    for key in DEFAULTS:
        value=getattr(args,key)
        if value is None: continue
        if key in ('init_from','data_from') and value: value=str(Path(value).resolve())
        if old and key not in RUNTIME and value!=config[key]:
            raise ValueError(f'Existing run requires {key}={config[key]}; choose a new run-dir for changes')
        config[key]=value
    for key,value in config.items():
        if key in ('device','init_from','data_from'): continue
        zero_allowed=('seed','max_updates_per_stage','repair_updates','plateau_patience','stop_on_plateau')
        lower=0 if key in zero_allowed else (1e-12 if key in ('lr','dynamics_lr') else 1)
        if not math.isfinite(value) or value<lower: raise ValueError(f'Invalid {key}')
    for key in ('init_from','data_from'):
        if config[key]: config[key]=str(Path(config[key]).resolve())
    if config['horizon']!=10: raise ValueError('This curriculum requires horizon=10')
    if config['max_rank']<3 or config['max_rank']>20: raise ValueError('max-rank must be 3..20')
    if config['batch_size']<2 or config['validation_samples']<2: raise ValueError('batch and validation samples must be >=2')
    if config['head_batch_size']<2 or config['head_families']<30: raise ValueError('Need head-batch-size >=2 and head-families >=30')
    if config['stop_on_plateau'] not in (0,1): raise ValueError('stop-on-plateau must be 0 or 1')
    if config['init_from'] and not config['data_from']:
        raise ValueError('--init-from requires --data-from to preserve the original held-out split')
    if config['width']%config['heads'] or (config['width']//config['heads'])%4:
        raise ValueError('2D RoPE requires width/heads divisible by four')
    atomic_json(path,dict(pipeline_version=3,config=config))
    return config


def state(root):
    path=root/'progress.json'
    result=json.loads(path.read_text()) if path.exists() else dict(status='running',stages={})
    result['stages']={k:v for k,v in result['stages'].items() if k in ('data',*STAGES,'audit')}
    return result


def update_status(root,stage,**values):
    if values.get('status')=='passed': values.setdefault('stalled',False)
    status=state(root);status['stages'].setdefault(stage,{}).update(values)
    status['status']=('complete' if stage=='audit' and values.get('status')=='passed'
                      else values['status'] if values.get('status') in ('paused','failed') else 'running')
    atomic_json(root/'progress.json',status);progress_plot(root,status)


def audit_data(data,samples=64):
    failures=[]
    for split,part in data.splits.items():
        coverage=data.coverage[split]
        if any(coverage['kinds'][k]==0 for k in KINDS): failures.append(split+': missing board family')
        if not coverage['terminal_roots'] or any(n==0 for n in coverage['actions']): failures.append(split+': incomplete head/action coverage')
        seq=part['sequence'];valid=seq['valid'].bool()
        if (valid[:,1:] & ~valid[:,:-1]).any(): failures.append(split+': discontinuous padding')
        if (valid[:,1:] & seq['dones'][:,:-1]).any(): failures.append(split+': transition after terminal')
        if not torch.equal(seq['states'][:,1:][valid[:,1:]],seq['next_states'][:,:-1][valid[:,1:]]):
            failures.append(split+': broken sequence')
        q=seq['chance_probs'][valid]
        if not torch.allclose(q.sum(-1),torch.ones(len(q)),atol=1e-6): failures.append(split+': probability normalization')
        point=data.sample(split,min(samples,len(part['flat']['actions'])),'cpu',np.random.default_rng(7123))
        for i in range(len(point['actions'])):
            after,p,boards,masks,dones=transition(point['states'][i].numpy(),int(point['actions'][i]))
            event=int(point['events'][i])
            ok=(np.array_equal(after,point['afterstates'][i].numpy()) and
                np.allclose(p,point['chance_probs'][i].numpy(),atol=1e-7) and p[event]>0 and
                np.array_equal(boards[event],point['next_states'][i].numpy()) and
                np.array_equal(masks,point['branch_masks'][i].numpy()) and
                np.array_equal(dones,point['branch_dones'][i].numpy()) and
                bool(dones[event])==bool(point['dones'][i]))
            if not ok: failures.append(split+': oracle label mismatch');break
    return dict(passed=not failures,failures=failures,coverage=data.coverage,data_sha256=data.digest,
                note='All sequence boundaries and probability sums checked; oracle labels checked on a fixed sample.')


def prepare(root,cfg):
    initial=initializer(root,cfg)
    copy_base_data(root,cfg)
    def progress(done,total):
        print(json.dumps(dict(stage='data',families=done,target=total)),flush=True)
        update_status(root,'data',status='generating',generated=done)
    generate(root/'data',cfg['families'],cfg['horizon'],cfg['seed'],cfg['workers'],cfg['max_rank'],progress)
    data=SyntheticData(root/'data')
    if initial:
        source=read_checkpoint(initial)
        source_digest=source.get('base_dataset_sha256',source.get('dataset_sha256'))
        if source_digest!=data.digest: raise ValueError('Warm start must reuse its original base dataset and split')
    report=audit_data(data);atomic_json(root/'data'/'verification.json',report)
    coverage_plot(data.coverage,root/'data'/'coverage.png')
    update_status(root,'data',status='passed' if report['passed'] else 'failed',
                  generated=cfg['families'],failures=report['failures'])
    if not report['passed']: raise ValueError('Data coverage/labels failed: '+str(report['failures']))
    if initial and source.get('objective_version')==3 and not (root/'head_data').exists():
        # A v3 source has already seen its boundary train split: preserve that
        # entire corpus as well, instead of silently redefining its holdout.
        previous=Path(cfg['data_from']).resolve().parent/'head_data'
        if not previous.exists(): raise ValueError('A v3 warm start also needs the original sibling head_data directory')
        temporary=root/'.head-data-import'
        if temporary.exists(): shutil.rmtree(temporary)
        shutil.copytree(previous,temporary);temporary.replace(root/'head_data')
    owners=prepare_boundaries(root/'head_data',data,cfg['head_families'],cfg['seed'],cfg['max_rank'],cfg['workers'],
        lambda done,total:update_status(root,'data',status='generating',head_generated=done,head_target=total))
    data=CurriculumData(data,root/'head_data',owners)
    if initial and source.get('objective_version')==3 and source['dataset_sha256']!=data.digest:
        raise ValueError('A v3 warm start must preserve both original data corpora')
    failures=[]
    from .verification import reference
    for split,part in data.heads.items():
        for board,mask,done in zip(part['boards'],part['masks'],part['dones']):
            truth=reference(board.numpy())
            if bool(done)!=truth['done'] or not np.array_equal(mask.numpy(),truth['mask']):
                failures.append(split+': boundary oracle label mismatch');break
    atomic_json(root/'head_data'/'verification.json',dict(passed=not failures,failures=failures,
        data_sha256=data.digest,coverage={s:dict(boards=len(p['boards']),terminals=int(p['dones'].sum())) for s,p in data.heads.items()}))
    if failures: raise ValueError(str(failures))
    # Predecessor checks bind BOTH immutable corpora, not just base transitions.
    report['data_sha256']=data.digest;report['base_sha256']=data.base_digest
    atomic_json(root/'data'/'verification.json',report)
    update_status(root,'data',status='passed',generated=cfg['families'],head_generated=cfg['head_families'],failures=[])
    from .plots import head_coverage_plot
    head_coverage_plot(data,root/'head_data'/'coverage.png')
    return data


def verify_predecessor(root,stage,data):
    if stage=='tokenizer':
        report=json.loads((root/'data'/'verification.json').read_text())
        if not report['passed'] or report['data_sha256']!=data.digest: raise ValueError('Data validation must pass first')
        return None
    previous=STAGES[STAGES.index(stage)-1]
    report_path=root/previous/'validation.json';checkpoint=root/previous/'passed.pt'
    if not report_path.exists() or not checkpoint.exists(): raise ValueError(f'{previous} must pass before {stage}')
    report=json.loads(report_path.read_text())
    if (not report['passed'] or report.get('consecutive_passes',0)<2 or report['data_sha256']!=data.digest
            or report['checkpoint_sha256']!=file_digest(checkpoint)):
        raise ValueError(f'{previous} passing report does not match checkpoint/data')
    return checkpoint


def load_history(path,iteration):
    if not path.exists(): return []
    rows=[]
    for line in path.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: break
        if row['iteration']<=iteration: rows.append(row)
    # Remove any log tail newer than the atomic checkpoint after interrupted IO.
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def train_stage(root,cfg,data,stage,minimum_updates=0):
    parent=verify_predecessor(root,stage,data)
    directory=root/stage;directory.mkdir(exist_ok=True)
    device=setup(cfg['seed'],cfg['device'])
    last=directory/'last.pt';source=read_checkpoint(last) if last.exists() else None
    if source:
        if source.get('dataset_sha256')!=data.digest or source.get('pipeline_stage')!=stage:
            raise ValueError('Checkpoint stage or data mismatch')
        model=model_from_checkpoint(source,device)
    elif parent:
        model=model_from_checkpoint(read_checkpoint(parent),device)
    else:
        model=load_initial_model(root,cfg,device)
    phase='tokenizer' if stage=='tokenizer' else 'world';model.set_phase(phase)
    optimizer=optimizer_for(model,cfg,stage)
    enhanced=getattr(data,'optimized',False)
    iteration=0;streak=0;best=-float('inf');last_validation=-1;report=None
    if source:
        restore_checkpoint(source,model,optimizer,restore_rng=True)
        iteration=source['iteration'];streak=source.get('consecutive_passes',0)
        best=source['best_metric'];last_validation=source.get('last_validation',-1)
    history=load_history(directory/'metrics.jsonl',iteration)
    if (directory/'validation.json').exists(): report=json.loads((directory/'validation.json').read_text())
    def persist():
        configuration={**cfg,'phase':phase,'data':str((root/'data').resolve()),'architecture':'latent_transformer'}
        save_checkpoint(last,model,optimizer,iteration,'latent_dreamer',configuration,best,
            dict(dataset_sha256=data.digest,base_dataset_sha256=getattr(data,'base_digest',data.digest),
                 objective_version=3,pipeline_stage=stage,consecutive_passes=streak,last_validation=last_validation))
    def record(row):
        history.append(row)
        with (directory/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(stage=stage,**row),allow_nan=False),flush=True)
    plateau=plateau_state(history,iteration,cfg['plateau_patience'])
    while True:
        if iteration!=last_validation and (iteration==0 or iteration%cfg['eval_every']==0):
            metrics=evaluate(model,data,stage,cfg['validation_samples'],cfg['batch_size'],cfg['seed']+9000000)
            if stage=='tokenizer' and iteration==0 and cfg['init_from'] and source is None and metrics['reconstruction']>=.995:
                second=evaluate(model,data,stage,cfg['validation_samples'],cfg['batch_size'],cfg['seed']+9000001)
                metrics['reconstruction']=min(metrics['reconstruction'],second['reconstruction'])
                # Two independent acceptance probes preserve a good imported E/D.
                streak=1 if metrics['reconstruction']>=.995 else 0
            report=gates(stage,metrics,enhanced);last_validation=iteration
            streak=streak+1 if report['passed'] else 0
            score=min((row['value'] if row['direction']=='min' else 1-row['value'])
                      if row['value'] is not None else -1 for row in report['checks'].values())
            improved=score>best;best=max(best,score);persist()
            report.update(iteration=iteration,stage=stage,metrics=metrics,consecutive_passes=streak,
                data_sha256=data.digest,world_sha256=world_fingerprint(model.world),checkpoint_sha256=file_digest(last))
            atomic_json(directory/'validation.json',report)
            if improved: shutil.copyfile(last,directory/'best.pt')
            train_metrics=evaluate(model,data,stage,cfg['train_probe_samples'],cfg['batch_size'],
                cfg['seed']+9300000,'train',False) if enhanced else {}
            record(dict(iteration=iteration,validation=metrics,train_probe=train_metrics,
                gate_margin=gate_margin(report),passed=report['passed'],failures=report['failures']))
            training_plots(directory,history,stage,report)
            reconstruction_plot(model,data,'validation',directory/'reconstruction.png')
            update_status(root,stage,status='training',iteration=iteration,failures=report['failures'])
            plateau=plateau_state(history,iteration,cfg['plateau_patience'])
        plateau['updates_without_improvement']=iteration-plateau['last_improvement']
        plateau['stalled']=bool(cfg['plateau_patience'] and plateau['updates_without_improvement']>=cfg['plateau_patience'])
        if report and report['passed'] and streak>=2 and iteration>=minimum_updates:
            # A report from an older update is never sufficient to advance.
            if report['checkpoint_sha256']==file_digest(last) and report['iteration']==iteration:
                shutil.copyfile(last,directory/'passed.pt')
                update_status(root,stage,status='passed',iteration=iteration,failures=[])
                return True
        if plateau['stalled'] and cfg['stop_on_plateau']:
            persist();update_status(root,stage,status='paused',iteration=iteration,failures=(report or {}).get('failures',[]),**plateau)
            print(json.dumps(dict(stage=stage,event='plateau_stop',**plateau)),flush=True);return False
        limit=cfg['max_updates_per_stage']
        if limit and iteration>=limit:
            training_plots(directory,history,stage,report)
            update_status(root,stage,status='paused',iteration=iteration,failures=(report or {}).get('failures',[]),**plateau)
            return False
        model.train();optimizer.zero_grad(set_to_none=True)
        repairing=set_repair_phase(optimizer,stage,iteration,cfg)
        # Draw seed from saved NumPy global RNG: exact CPU resume across stages.
        rng=np.random.default_rng(np.random.randint(0,2**32-1))
        started=time.perf_counter()
        if stage=='tokenizer':
            boards=data.tokenizer_boards('train',cfg['batch_size'],device,rng)
            logits=model.world.decode(model.world.encode(boards))
            loss=F.cross_entropy(logits.flatten(0,1),boards.long().flatten())
            loss.backward();value=float(loss.detach());branch_value=0.;details={}
        else:
            horizon={'world1':1,'world3':3,'world10':10}[stage]
            batch=data.sample_sequence('train',cfg['batch_size'],horizon,device,rng)
            heads=data.sample_heads('train',cfg['head_batch_size'],device,rng)
            loss,details=supervised_world_loss(model,batch,heads,enhanced=enhanced)
            loss.backward();value=float(loss.detach());branch_value=0.
            branch=(data.sample_prior if enhanced else data.sample)('train',cfg['branch_batch_size'],device,rng)
            for branch_loss in full_branch_loss_chunks(model,branch,cfg['branch_chunk_size'],enhanced=enhanced,metrics=details):
                branch_value+=float(branch_loss.detach());branch_loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],100.)
        if not math.isfinite(value+branch_value) or not bool(torch.isfinite(norm)):
            raise FloatingPointError('Non-finite supervised loss/gradient; last checkpoint preserved')
        optimizer.step();iteration+=1
        if iteration%cfg['plot_every']==0 or iteration%cfg['eval_every']==0 or (limit and iteration==limit):
            persist();record(dict(iteration=iteration,loss=value+branch_value,sequence_loss=value,
                branch_loss=branch_value,grad_norm=float(norm),seconds=time.perf_counter()-started,
                repair_phase=repairing,**{k:float(v) for k,v in details.items()}))
            training_plots(directory,history,stage,report)
            update_status(root,stage,status='training',iteration=iteration,failures=(report or {}).get('failures',[]),**plateau)


def game_job(job):
    from gym2048_env import Gym2048Env
    from common.models import preprocess_observation
    from .verification import generate_latent_game,verify_latent_game,save_game_report
    checkpoint,seed,max_steps,output=job
    setup(seed,'cpu');model=model_from_checkpoint(read_checkpoint(checkpoint)).eval()
    env=Gym2048Env(seed=seed);board,_=env.reset(seed=seed);env.close()
    trace=generate_latent_game(model.world,preprocess_observation(board),seed,max_steps)
    report=verify_latent_game(model.world,trace,.01,.05)
    save_game_report(report,Path(output)/f'game_{seed}.json')
    torch.save(trace,Path(output)/f'game_{seed}.latent.pt')
    return {k:v for k,v in report.items() if k!='trajectory'}


def audit(root,cfg,data,split='validation'):
    directory=root/'audit';directory.mkdir(exist_ok=True)
    checkpoint=root/'world10'/'passed.pt'
    report_path=root/'world10'/'validation.json'
    if not checkpoint.exists() or not report_path.exists(): raise ValueError('world10 must pass before whole-game audit')
    before=json.loads(report_path.read_text())
    if (not before['passed'] or before['consecutive_passes']<2 or before['data_sha256']!=data.digest
            or before['checkpoint_sha256']!=file_digest(checkpoint)):
        raise ValueError('world10 verification does not match passing checkpoint')
    device=setup(cfg['seed'],cfg['device']);model=model_from_checkpoint(read_checkpoint(checkpoint),device)
    metrics=evaluate(model,data,'world10',cfg['validation_samples'],cfg['batch_size'],cfg['seed']+9100000,split)
    checked=gates('world10',metrics,getattr(data,'optimized',False))
    # CPU workers hold separate world copies; they never mutate the GPU model.
    seed=cfg['seed']+(4000000 if split=='validation' else 5000000)
    output=directory/split/f'update_{before["iteration"]:08d}';output.mkdir(parents=True,exist_ok=True)
    with GamePool(cfg['workers']) as pool:
        games=pool.map(game_job,[(str(checkpoint),seed+i,cfg['audit_max_steps'],str(output)) for i in range(cfg['audit_games'])])
    whole_games=all(g['passed'] for g in games)
    checked['checks']['whole_games']=dict(value=sum(g['passed'] for g in games)/len(games),direction='min',threshold=1.,passed=whole_games)
    checked['passed']=checked['passed'] and whole_games
    if not whole_games: checked['failures'].append('whole_games')
    passed=checked['passed']
    result=dict(passed=passed,split=split,iteration=before['iteration'],metrics=metrics,gates=checked,games=games,
        failures=checked['failures'],
        data_sha256=data.digest,world_sha256=world_fingerprint(model.world),checkpoint_sha256=file_digest(checkpoint))
    atomic_json(directory/f'{split}.json',result)
    training_plots(output,[dict(iteration=before['iteration'],validation=metrics)],'audit',checked)
    # Whole-game outcome is explicit in progress even if diagnostic gates passed.
    if split=='validation': update_status(root,'audit',status='passed' if passed else 'failed',iteration=before['iteration'],failures=result['failures'])
    print(json.dumps(dict(stage='audit',split=split,passed=passed,failures=result['failures'],games_passed=sum(g['passed'] for g in games))),flush=True)
    return result


def run(root,cfg,data):
    for stage in STAGES:
        if not train_stage(root,cfg,data,stage): return False
    while True:
        report=audit(root,cfg,data)
        if report['passed']: break
        # An audit failure must trigger real optimizer updates, not repeated
        # evaluation of the same passing short-rollout checkpoint.
        minimum=report['iteration']+cfg['eval_every']
        if not train_stage(root,cfg,data,'world10',minimum_updates=minimum): return False
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','data',*STAGES,'validate','verify','status'))
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--stage',choices=STAGES,default='world10')
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    for key,value in DEFAULTS.items(): parser.add_argument('--'+key.replace('_','-'),type=type(value))
    args=parser.parse_args(argv);root=Path(args.run_dir).resolve()
    if args.command=='status':
        print(json.dumps(state(root),indent=2));return True
    with lock_run(root):
        cfg=settings(args)
        if args.command in ('run','data'): data=prepare(root,cfg)
        else: data=CurriculumData(SyntheticData(root/'data'),root/'head_data')
        if args.command=='data': return True
        if args.command=='run': ok=run(root,cfg,data)
        elif args.command in STAGES: ok=train_stage(root,cfg,data,args.command)
        elif args.command=='verify': ok=audit(root,cfg,data,args.split)['passed']
        else:
            checkpoint=root/args.stage/'last.pt'
            if not checkpoint.exists(): raise ValueError('No checkpoint for requested stage')
            model=model_from_checkpoint(read_checkpoint(checkpoint),setup(cfg['seed'],cfg['device']))
            metrics=evaluate(model,data,args.stage,cfg['validation_samples'],cfg['batch_size'],cfg['seed']+9200000,args.split)
            report=gates(args.stage,metrics,True);report.update(metrics=metrics,split=args.split,checkpoint_sha256=file_digest(checkpoint))
            output=root/args.stage/f'manual_{args.split}';output.mkdir(exist_ok=True)
            atomic_json(output/'verification.json',report)
            training_plots(output,[dict(iteration=read_checkpoint(checkpoint)['iteration'],validation=metrics)],args.stage,report)
            print(json.dumps(report,indent=2));ok=report['passed']
        if not ok: raise SystemExit(2)
        return True


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. The last atomic checkpoint is preserved; rerun the same command to continue.',flush=True)
        raise SystemExit(130)
