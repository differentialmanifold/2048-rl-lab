# PPO Afterstate

**English** | [简体中文](ppo-afterstate.zh-CN.md)

The [implementation](../algorithms/ppo_afterstate.py) separates a move into deterministic sliding/merging and a random tile spawn. A shared CNN2×2 or ViT scores the four candidate afterstates, masks illegal actions and selects a direction before the tile is spawned.

The value head estimates each afterstate's expected return, including the following spawn reward. The decision-state baseline averages these values under the policy. Training uses clipped PPO with TD(10, 0.5) and γ=0.999. Rewards and values are divided by 128 internally; reported returns use raw spawn mass.

## Training

```sh
.venv/bin/python -m algorithms.ppo_afterstate \
  --architecture cnn2x2 --seed 0 --iterations 20000 \
  --device auto --workers 8 \
  --save-dir checkpoints/ppo_afterstate_cnn2x2_td_seed0_klfix
```

Defaults: 8 games per iteration, up to 4 minibatch passes, batch 256, learning rate 3e-4, clip 0.2, entropy coefficient 0.01, value coefficient 0.5 and KL threshold 0.02. Validation and plots run every 25 iterations, using 10 fixed-seed games. CPU workers collect games; the parent updates on the selected device.

## Resume and play

```sh
.venv/bin/python -m algorithms.ppo_afterstate \
  --resume checkpoints/ppo_afterstate_cnn2x2_td_seed0_klfix/last.pt \
  --iterations 30000 --workers 8

.venv/bin/python play.py --agent ppo_afterstate --auto

.venv/bin/python evaluate.py --agent ppo_afterstate \
  --checkpoint pretrained/ppo_afterstate_cnn2x2.pt \
  --episodes 100 --seed 2000000 --workers 8 \
  --output experiments/ppo_afterstate_test.json
```

`--iterations` is a cumulative target. Resume restores the model, optimizer and RNG. Outputs are `last.pt`, `best.pt` selected by mean real-game validation return, `metrics.jsonl` and `training.png`.

## Recorded result

2026-10-07: CNN2×2 completed 20,000 iterations. The best checkpoint at iteration 16,575 averages 6,930.7 moves and 15,270.4 spawn return over 10 validation games, reaching 8192 in 90% and 16384 in 20%. These are checkpoint-selection validation results. Its inference weights are bundled.

![PPO Afterstate · CNN2×2](../assets/ppo_afterstate_cnn2x2.png)
