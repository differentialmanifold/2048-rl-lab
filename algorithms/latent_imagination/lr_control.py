"""Checkpointed PPO learning-rate control; KL early stopping stays enabled."""
import math


class KLRateController:
    def __init__(self, mode, lr, minimum=1e-6, maximum=None, patience=5, state=None):
        maximum = lr if maximum is None else maximum
        if mode not in ('constant', 'adaptive_kl') or not 0 < minimum <= lr <= maximum or patience < 1:
            raise ValueError('Invalid learning-rate controller settings')
        self.spec = dict(mode=mode, minimum=minimum, maximum=maximum, patience=patience)
        self.lr, self.down, self.up, self.steps = lr, 0, 0, 0
        if state is not None:
            if state['spec'] != self.spec:
                raise ValueError('Saved LR controller settings differ; explicitly reset --lr to change them')
            self.lr, self.down, self.up, self.steps = (state[k] for k in ('lr', 'down', 'up', 'steps'))
            if not minimum <= self.lr <= maximum:
                raise ValueError('Saved learning rate is outside configured bounds')

    def state_dict(self):
        return dict(spec=self.spec, lr=self.lr, down=self.down, up=self.up, steps=self.steps)

    def apply(self, optimizer):
        for group in optimizer.param_groups:
            group['lr'] = self.lr

    def advance(self, metrics, target_kl):
        used = self.lr
        reason = 'constant' if self.spec['mode'] == 'constant' else 'hold'
        self.steps += 1
        if self.spec['mode'] == 'adaptive_kl':
            kl = metrics['rollout_kl']
            if not math.isfinite(kl) or target_kl <= 0:
                raise ValueError('Adaptive LR requires a finite rollout KL and positive target KL')
            # A minibatch stop before seeing one full rollout is also pressure:
            # otherwise final KL stays near its cap while data use collapses.
            pressure = (kl > 1.5 * target_kl or (metrics['update_stop_reason'] == 'target_kl'
                                                and metrics['effective_epochs'] < 1.))
            room = metrics['update_stop_reason'] == 'epochs_complete' and kl < .5 * target_kl
            self.down = self.down + 1 if pressure else 0
            self.up = self.up + 1 if room else 0
            if self.down >= self.spec['patience']:
                self.lr = max(self.spec['minimum'], self.lr * .8)
                self.down = 0
                reason = 'reduce_kl_pressure' if self.lr < used else 'minimum'
            elif self.up >= self.spec['patience']:
                self.lr = min(self.spec['maximum'], self.lr * 1.1)
                self.up = 0
                reason = 'increase_kl_room' if self.lr > used else 'maximum'
        return dict(learning_rate=used, next_learning_rate=self.lr,
                    lr_schedule=self.spec['mode'], lr_adjustment=reason)


def plot_control(logs):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in logs.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if 'learning_rate' in r]
    if not rows:return
    x=[r['iteration'] for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
    for ax,key,title in zip(axes,['learning_rate','effective_epochs','rollout_kl'],
                           ['Adaptive learning rate','Effective epochs','Final rollout KL']):
        ax.plot(x,[r[key] for r in rows]);ax.set(title=title,xlabel='PPO iteration');ax.grid(alpha=.2)
    axes[0].set_yscale('log')
    fig.savefig(logs.parent/'optimization.png',dpi=140);plt.close(fig)
