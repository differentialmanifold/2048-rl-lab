"""Generate latent-only games first, then independently audit decoded traces.

No oracle result, decoded board, or reference action mask feeds the rollout.
The verifier never repairs a prediction. A truncated or broken game cannot pass.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ..runtime import isolated_rng
from .diagnostics import raw_board


def reference(ranks, action=None):
    """Evaluation/training oracle, never called by generate_latent_game."""
    from board import Board
    from common.models import preprocess_observation
    board = Board(raw_board(ranks))
    mask = list(map(bool, board.can_move_dir))
    if action is None:
        return dict(mask=mask, done=not any(mask))
    lines = (board.matrix, board.matrix.T, board.matrix[:, ::-1], board.matrix.T[:, ::-1])[action]
    changed = board._move_lines(lines)
    return dict(afterstate=preprocess_observation(board.matrix).long().tolist(),
                changed=changed, mask=mask, done=not any(mask), spawn4_probability=Board.fourProbability)


def observed_event(after, following, changed):
    """33 semantic events: cell*2 + (rank-1), or 32 for no spawn."""
    after, following = np.asarray(after), np.asarray(following)
    different = np.flatnonzero(after != following)
    if not changed:
        return 32 if len(different) == 0 else None
    if len(different) != 1:
        return None
    cell = int(different[0])
    if after[cell] != 0 or following[cell] not in (1, 2):
        return None
    return 2 * cell + int(following[cell]) - 1


@torch.no_grad()
def generate_latent_game(world, initial_ranks, seed=0, max_steps=10000, actor=None):
    """Only encode at reset; recursive states/masks come exclusively from world.

    World interface: encode, state_heads, step_details. Decoding is deliberately
    absent, including when actor is used. Actor consumes latent states.
    """
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    device = next(world.parameters()).device
    initial = torch.as_tensor(initial_ranks, device=device).reshape(1, 16)
    with isolated_rng(device, seed):
        z = world.encode(initial)
        trace = dict(initial_ranks=initial.cpu().tolist()[0], initial_z=z.cpu(),
                     seed=seed, steps=[], stop_reason='max_steps')
        for _ in range(max_steps):
            try:
                terminal, legal = world.state_heads(z)
            except ValueError as error:
                trace.update(stop_reason='model_error', model_error=str(error))
                break
            mask = legal >= 0
            done = bool(terminal.sigmoid().item() >= .5)
            if done or not bool(mask.any()):
                trace['stop_reason'] = 'model_terminal' if done else 'no_predicted_legal_actions'
                break
            if actor is None:
                probabilities = mask.float() / mask.sum(-1, keepdim=True)
            else:
                probabilities = actor(z).masked_fill(~mask, -torch.inf).softmax(-1)
            action = torch.multinomial(probabilities, 1).squeeze(-1)
            try:
                details = world.step_details(z, action)
            except ValueError as error:
                trace.update(stop_reason='model_error', model_error=str(error), failed_action=int(action.item()))
                break
            tensors = [v for v in details.values() if torch.is_tensor(v)]
            if not all(bool(torch.isfinite(v).all()) for v in tensors):
                trace['stop_reason'] = 'non_finite_prediction'
                break
            trace['steps'].append(dict(action=int(action.item()),
                terminal=terminal.cpu(), legal=legal.cpu(),
                **{k: v.cpu() if torch.is_tensor(v) else v for k, v in details.items()}))
            z = details['next_z']
        # Check final model state even when the last allowed move ends the game.
        try:
            terminal, legal = world.state_heads(z)
        except ValueError as error:
            trace.update(stop_reason='model_error', model_error=str(error))
            terminal, legal = torch.zeros(1, device=device), torch.zeros(1, 4, device=device)
        trace['final_terminal'] = terminal.cpu()
        trace['final_legal'] = legal.cpu()
        if trace['stop_reason'] == 'max_steps':
            if bool(terminal.sigmoid().item() >= .5):
                trace['stop_reason'] = 'model_terminal'
            elif not bool((legal >= 0).any()):
                trace['stop_reason'] = 'no_predicted_legal_actions'
    return trace


@torch.no_grad()
def verify_latent_game(world, trace, probability_tolerance=1e-7, distribution_tolerance=None):
    """Post-hoc comparison on local decoded states AND initial-state anchored replay."""
    device = next(world.parameters()).device
    def decode(z):
        return world.decode(z.to(device)).argmax(-1).cpu().reshape(-1).tolist()
    current = decode(trace['initial_z'])
    initial_ok = current == trace['initial_ranks']
    anchored = list(trace['initial_ranks'])
    rows = []
    semantic_events = bool(getattr(world, 'semantic_events', False))
    for index, step in enumerate(trace['steps']):
        action = step['action']
        after, following = decode(step['afterstate']), decode(step['next_z'])
        expected = reference(current, action)
        next_reference = reference(following)
        event = observed_event(expected['afterstate'], following, expected['changed'])
        mask = (step['legal'] >= 0).reshape(-1).tolist()
        next_mask = (step['next_legal'] >= 0).reshape(-1).tolist()
        model_done = bool(step['next_terminal'].sigmoid().item() >= .5)
        sampled_event = int(step['event'].item())
        reward = float(step['reward'].item())
        expected_reward = None if event is None else (0 if event == 32 else (2 if event % 2 == 0 else 4))
        probabilities = step['probabilities'].reshape(-1).numpy()
        probability_ok = (np.isfinite(probabilities).all() and (probabilities >= 0).all()
                          and abs(probabilities.sum()-1) < 1e-5)
        chance_tv, support_ok = None, True
        if semantic_events:
            expected_probabilities = np.zeros(33)
            if expected['changed']:
                empty = np.flatnonzero(np.asarray(expected['afterstate']) == 0)
                expected_probabilities[empty*2] = (1-expected['spawn4_probability'])/len(empty)
                expected_probabilities[empty*2+1] = expected['spawn4_probability']/len(empty)
            else:
                expected_probabilities[32] = 1.
            chance_tv = float(.5*np.abs(probabilities-expected_probabilities).sum())
            support_ok = float(probabilities[expected_probabilities == 0].sum()) <= probability_tolerance
        anchored_after = reference(anchored, action) if anchored is not None else None
        anchored_event = (observed_event(anchored_after['afterstate'], following,
                          anchored_after['changed']) if anchored_after is not None else None)
        anchored_ok = (anchored_after is not None and current == anchored
                       and after == anchored_after['afterstate'] and anchored_event is not None)
        checks = dict(current_legal_mask=mask == expected['mask'],
            selected_action_legal=bool(expected['mask'][action]),
            current_terminal=(bool(step['terminal'].sigmoid().item() >= .5) == expected['done']),
            afterstate=after == expected['afterstate'], spawn_rule=event is not None,
            event_matches_spawn=(event == sampled_event if semantic_events else True),
            reward=(expected_reward is not None and abs(reward - expected_reward) < 1e-5),
            next_legal_mask=next_mask == next_reference['mask'],
            next_terminal=model_done == next_reference['done'], anchored_chain=anchored_ok,
            chance_normalized=bool(probability_ok), chance_support=support_ok,
            chance_distribution=(distribution_tolerance is None or chance_tv is None
                                 or chance_tv <= distribution_tolerance),
            sampled_event_supported=bool(0 <= sampled_event < len(probabilities)
                                         and probabilities[sampled_event] > 0))
        row = dict(step=index, action=action, action_name=('LEFT', 'UP', 'RIGHT', 'DOWN')[action],
            current=raw_board(current).tolist(), afterstate=raw_board(after).tolist(),
            expected_afterstate=raw_board(expected['afterstate']).tolist(),
            following=raw_board(following).tolist(), sampled_event=sampled_event,
            observed_event=event, reward=reward, expected_reward=expected_reward,
            reward_mean=float(step.get('reward_mean', step['reward']).item()),
            predicted_legal=mask, reference_legal=expected['mask'],
            next_predicted_legal=next_mask, next_reference_legal=next_reference['mask'],
            next_terminal_probability=float(step['next_terminal'].sigmoid().item()),
            next_reference_terminal=next_reference['done'],
            chance_probabilities=probabilities.tolist(), chance_tv=chance_tv,
            checks=checks, passed=all(checks.values()))
        rows.append(row)
        # Anchored chain becomes permanently invalid on any transition error.
        anchored = following if anchored_ok else None
        current = following
    final = reference(current)
    final_mask = (trace['final_legal'] >= 0).reshape(-1).tolist()
    final_done = bool(trace['final_terminal'].sigmoid().item() >= .5)
    final_checks = dict(terminal=final_done == final['done'],
        legal_mask=final_mask == final['mask'], actual_game_over=final['done'],
        natural_stop=trace['stop_reason'] in ('model_terminal', 'no_predicted_legal_actions'))
    errors = [dict(step=r['step'], checks=[k for k, ok in r['checks'].items() if not ok])
              for r in rows if not r['passed']]
    if not initial_ok:
        errors.insert(0, dict(step=0, checks=['initial_reconstruction']))
    if trace.get('model_error'):
        errors.append(dict(step=len(rows), checks=['model_error'], message=trace['model_error'],
                           action=trace.get('failed_action')))
    if not all(final_checks.values()):
        errors.append(dict(step=len(rows), checks=['final_'+k for k,v in final_checks.items() if not v]))
    return dict(seed=trace['seed'], steps=len(rows), stop_reason=trace['stop_reason'],
        probability_tolerance=probability_tolerance,
        distribution_tolerance=distribution_tolerance,
        initial=raw_board(trace['initial_ranks']).tolist(), initial_reconstruction=initial_ok,
        final=raw_board(current).tolist(), final_checks=final_checks,
        passed=initial_ok and not errors and all(final_checks.values()),
        first_error=errors[0] if errors else None, errors=errors,
        check_counts={k: sum(r['checks'][k] for r in rows) for k in rows[0]['checks']} if rows else {},
        spawn2=sum(r['observed_event'] is not None and r['observed_event'] < 32
                   and r['observed_event'] % 2 == 0 for r in rows),
        spawn4=sum(r['observed_event'] is not None and r['observed_event'] < 32
                   and r['observed_event'] % 2 == 1 for r in rows),
        trajectory=rows)


def save_game_report(report, path):
    """Self-contained local viewer, no network dependencies or executable data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    payload = json.dumps(report).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = '''<!doctype html><meta charset="utf-8"><title>Latent 2048 replay audit</title>
<style>body{font:16px system-ui;margin:30px;background:#faf8ef;color:#463e36}button,input{font:inherit;margin:6px;padding:6px}#boards{display:flex;flex-wrap:wrap;gap:22px}table{border-spacing:5px;background:#b9aca0;padding:5px}td{background:#ede0cc;width:65px;height:60px;text-align:center;font-weight:bold}pre{white-space:pre-wrap}.fail{color:#ad2626}.ok{color:#23633a}</style>
<h1>潜空间 2048 · 逐步核验</h1><p id="summary"></p>
<button onclick="show(at-1)">上一步</button><button onclick="show(at+1)">下一步</button>
<button onclick="show(data.first_error?.step??0)">首个错误</button><input id="slider" type="range" min="0" oninput="show(+this.value)"><span id="step"></span>
<div id="boards"></div><pre id="checks"></pre><details><summary>随机事件概率</summary><pre id="probs"></pre></details>
<details><summary>整局结束检查</summary><pre id="final"></pre></details>
<script type="application/json" id="data">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);let at=0;
document.getElementById('summary').textContent=`整局校验 ${data.passed?'通过':'未通过'}；${data.steps} 步；停止原因 ${data.stop_reason}。校验结果没有反馈给模型。`;
document.getElementById('final').textContent=JSON.stringify({checks:data.final_checks,errors:data.errors},null,2);
document.getElementById('slider').max=Math.max(0,data.steps-1);
function board(title,m){const d=document.createElement('div'),h=document.createElement('h3'),t=document.createElement('table');h.textContent=title;d.append(h,t);m.forEach(row=>{const tr=t.insertRow();row.forEach(v=>{tr.insertCell().textContent=v||'';});});return d;}
function show(i){at=Math.max(0,Math.min(i,data.steps-1));const root=document.getElementById('boards');root.replaceChildren();const r=data.trajectory[at];if(!r){root.append(board('初始棋盘',data.initial));document.getElementById('checks').textContent=JSON.stringify(data.final_checks,null,2);return;}
document.getElementById('slider').value=at;document.getElementById('step').textContent=`第 ${at+1}/${data.steps} 步 · ${r.action_name}`;
root.append(board('当前：解码',r.current),board('Afterstate：模型',r.afterstate),board('Afterstate：规则参考',r.expected_afterstate),board('下一状态：模型',r.following));
const c=document.getElementById('checks');c.className=r.passed?'ok':'fail';c.textContent=JSON.stringify({checks:r.checks,legal_model:r.predicted_legal,legal_reference:r.reference_legal,next_legal_model:r.next_predicted_legal,next_legal_reference:r.next_reference_legal,reward:r.reward,reward_mean:r.reward_mean,event:r.sampled_event},null,2);document.getElementById('probs').textContent=JSON.stringify(r.chance_probabilities,null,2);}
show(0);</script>'''
    path.with_suffix('.html').write_text(template.replace('PAYLOAD', payload))
