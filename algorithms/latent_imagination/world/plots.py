"""Atomic, headless PNGs for data coverage and gated training progress."""
import json
from pathlib import Path
import numpy as np


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save(fig,path):
    plt=pyplot();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.png');fig.savefig(tmp,dpi=130);plt.close(fig);tmp.replace(path)


def coverage_plot(coverage,path):
    plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for split,row in coverage.items():
        for ax,key,title in zip(axes.flat,['kinds','empty_cells','tile_ranks','actions'],
                ['Board families','Empty cells per root','Tile rank coverage','Action coverage']):
            values=row[key];labels=list(values) if isinstance(values,dict) else list(range(len(values)))
            y=list(values.values()) if isinstance(values,dict) else values
            ax.plot(labels,y,'o-',label=split);ax.set_title(title);ax.set_yscale('symlog',linthresh=1)
            ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Synthetic data coverage — counts, not model accuracy')
    save(fig,path)


def progress_plot(root,status):
    plt=pyplot();stages=list(status.get('stages', {})) or ['data','tokenizer','world1','world3','world10','audit']
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    colors={'passed':'#228b61','training':'#d49323','pending':'#c7cbd1','failed':'#bc4545','generating':'#d49323','paused':'#d49323'}
    for i,stage in enumerate(stages):
        row=status.get('stages',{}).get(stage,{});state=row.get('status','pending')
        ax.barh(i,1,color=colors.get(state,'#c7cbd1'))
        detail=f'{stage}: {state}  |  update {row.get("iteration",0)}'
        if stage=='data': detail=f'data: {state}  |  {row.get("generated",0)} families'
        if stage=='data' and 'head_generated' in row: detail+=f'  |  {row["head_generated"]} boundary families'
        if row.get('stalled'): detail+='  |  PLATEAU'
        if row.get('failures'): detail+='  |  failed: '+', '.join(row['failures'][:3])
        ax.text(.02,i,detail,va='center',fontsize=10,color='black')
    ax.set_xlim(0,1);ax.set_xticks([]);ax.set_yticks([]);ax.invert_yaxis()
    ax.set_title('2048 world + policy pipeline — '+status.get('status','running'))
    save(fig,Path(root)/'progress.png')


def training_plots(directory,history,stage,report=None):
    if not history: return
    plt=pyplot()
    if stage=='tokenizer':
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        train=[r for r in history if 'loss' in r];val=[r for r in history if 'validation' in r]
        axes[0].plot([r['iteration'] for r in train],[r['loss'] for r in train],label='reconstruction loss')
        axes[0].set_title('Supervised reconstruction objective')
        axes[1].plot([r['iteration'] for r in val],[r['validation']['reconstruction'] for r in val],label='whole-board accuracy')
        axes[1].axhline(.995,color='black',ls=':',label='gate >= .995');axes[1].set_ylim(-.02,1.03)
        axes[1].set_title('Held-out validation, all 16 cells must match')
        for ax in axes: ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Optimizer updates')
        fig.suptitle(f'tokenizer · update {history[-1]["iteration"]} · '+('PASS' if report and report['passed'] else 'NOT PASSED'))
        save(fig,Path(directory)/'training.png')
        if report: gate_plot(directory,stage,report)
        return
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    train=[r for r in history if 'loss' in r]
    axes[0,0].plot([r['iteration'] for r in train],[r['loss'] for r in train],label='training loss')
    axes[0,0].set_yscale('symlog',linthresh=1);axes[0,0].set_title('Supervised objective')
    val=[r for r in history if 'validation' in r]
    for key in ['reconstruction','afterstate','next_state','all_branch_accuracy','legal','terminal_recall','encoded_next_legal','head_agreement']:
        rows=[r for r in val if key in r['validation']]
        if rows: axes[0,1].plot([r['iteration'] for r in rows],[r['validation'][key] for r in rows],label=key)
    axes[0,1].axhline(.995,color='black',ls=':',label='reconstruction gate .995')
    axes[0,1].set_ylim(-.02,1.03);axes[0,1].set_title('Held-out validation accuracy')
    for key in ['prior_tv','decoded_branch_tv','decoded_invalid_mass','terminal_false_positive']:
        rows=[r for r in val if key in r['validation']]
        if rows: axes[1,0].plot([r['iteration'] for r in rows],[r['validation'][key] for r in rows],label=key)
    axes[1,0].axhline(.03,color='black',ls=':',label='prior TV gate .03')
    axes[1,0].set_ylim(-.02,1.03);axes[1,0].set_title('Probability and terminal errors (lower is better)')
    if val:
        metrics=val[-1]['validation'];depths=sorted(map(int,metrics.get('depth_samples',{})))
        for key in ['afterstate','next_state','prefix']:
            if depths: axes[1,1].plot(depths,[metrics[f'step_{d}_{key}'] for d in depths],'o-',label=key)
    axes[1,1].set_ylim(-.02,1.03);axes[1,1].set_title('Latest recurrent rollout (true actions/events)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7)
        ax.set_xlabel('Prediction depth' if ax==axes[1,1] else 'Optimizer updates in this stage')
    failures=report.get('failures',[]) if report else []
    fig.suptitle(f'{stage} · update {history[-1]["iteration"]} · '+('PASS' if report and report['passed'] else 'NOT PASSED')+
                 ('\nFailed: '+', '.join(failures) if failures else ''))
    save(fig,Path(directory)/'training.png')
    if report: gate_plot(directory,stage,report)
    if any('train_probe' in r and r['train_probe'] for r in val):
        diagnosis_plot(directory,history,stage)


def diagnosis_plot(directory,history,stage):
    plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    val=[r for r in history if 'validation' in r];train=[r for r in history if 'loss' in r]
    for key in ('legal','terminal_recall','encoded_next_legal'):
        for source,style in [('validation','-'),('train_probe','--')]:
            rows=[r for r in val if key in r.get(source,{})]
            if rows: axes[0,0].plot([r['iteration'] for r in rows],[r[source][key] for r in rows],style,label=f'{source}: {key}')
    axes[0,0].set_title('Train / held-out generalization');axes[0,0].set_ylim(-.02,1.03)
    for key in ('generated_next_legal','encoded_next_legal','head_agreement','root_head_agreement'):
        rows=[r for r in val if key in r['validation']]
        if rows: axes[0,1].plot([r['iteration'] for r in rows],[r['validation'][key] for r in rows],label=key)
    axes[0,1].set_title('Matched encoded / generated readouts');axes[0,1].set_ylim(-.02,1.03)
    for key in ('noop_invalid_mass','changed_invalid_mass','terminal_false_positive'):
        rows=[r for r in val if key in r['validation']]
        if rows: axes[1,0].plot([r['iteration'] for r in rows],[r['validation'][key] for r in rows],label=key)
    axes[1,0].axhline(.01,color='black',ls=':');axes[1,0].set_title('Conditional errors; gate <= 1%')
    for key in ('encoded_head_nll','head_loss','head_alignment_loss','prior_kl','changed_nll','branch_prior_ce'):
        rows=[r for r in train if key in r]
        if rows: axes[1,1].plot([r['iteration'] for r in rows],[r[key] for r in rows],label=key)
    axes[1,1].set_title('Separated learning objectives');axes[1,1].set_yscale('symlog',linthresh=.01)
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.set_xlabel('Optimizer updates')
        if ax.lines: ax.legend(fontsize=6)
    fig.suptitle(stage+' — curriculum v3 diagnostics')
    save(fig,Path(directory)/'diagnostics.png')


def head_coverage_plot(data,path):
    import torch
    plt=pyplot();fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for split,part in data.heads.items():
        axes[0].plot(['live','terminal'],torch.bincount(part['dones'].long(),minlength=2).tolist(),'o-',label=split)
        axes[1].plot(range(5),torch.bincount(part['masks'].sum(-1),minlength=5).tolist(),'o-',label=split)
    for ax in axes: ax.set_yscale('symlog',linthresh=1);ax.grid(alpha=.2);ax.legend()
    axes[0].set_title('Boundary + base head examples');axes[1].set_title('Number of legal actions per board')
    save(fig,path)


def gate_plot(directory,stage,report):
    plt=pyplot()
    checks=report['checks'];fig,ax=plt.subplots(figsize=(12,max(3,len(checks)*.3)),constrained_layout=True)
    for i,(key,row) in enumerate(checks.items()):
        value=row['value'];ax.barh(i,1,color='#228b61' if row['passed'] else '#bc4545',alpha=.3)
        ax.text(.01,i,f'{key}: {value if value is None else f"{value:.4f}"}  {">=" if row["direction"]=="min" else "<="} {row["threshold"]}',va='center')
    ax.set_yticks([]);ax.set_xticks([]);ax.invert_yaxis();ax.set_title(f'{stage}: validation gates')
    save(fig,Path(directory)/'validation.png')


def reconstruction_plot(model,data,split,path):
    import torch
    plt=pyplot();device=next(model.parameters()).device
    boards=data.splits[split]['boards'][:4].to(device)
    with torch.no_grad(): pred=model.world.decode(model.world.encode(boards)).argmax(-1).cpu().numpy()
    truth=boards.cpu().numpy();fig,axes=plt.subplots(len(boards),2,figsize=(7,3*len(boards)),squeeze=False,constrained_layout=True)
    for row in range(len(boards)):
        for col,values in enumerate((truth[row],pred[row])):
            ax=axes[row,col];grid=values.reshape(4,4)
            ax.imshow(grid,vmin=0,vmax=16,cmap='YlOrBr')
            for i in range(4):
                for j in range(4):
                    rank=int(grid[i,j]);ax.text(j,i,str(2**rank) if rank else '.',ha='center',va='center',fontsize=8)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Truth' if col==0 else 'Decoded prediction')
    save(fig,path)
