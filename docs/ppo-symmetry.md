# PPO · D4 augmentation

**English** | [简体中文](ppo-symmetry.zh-CN.md)

[PPO](../algorithms/ppo.py) optionally adds rotation/reflection consistency with `--symmetry d4`. Each training sample receives one of the eight D4 transforms. The transformed policy is mapped back to the original action coordinates; its distribution and value are trained to match detached predictions on the original board.

The original rollout supplies PPO ratios, advantages and TD targets. Transformed boards enter only the auxiliary losses. Inference uses a single forward pass, so direction preferences can remain.

## Train

```sh
.venv/bin/python -m algorithms.ppo \
  --architecture cnn2x2 --seed 0 --iterations 20000 \
  --symmetry d4 --symmetry-policy-coef 0.1 --symmetry-value-coef 0.1 \
  --save-dir checkpoints/ppo_cnn2x2_d4_seed0
```

The same option supports ViT. `--symmetry none` is ordinary PPO. Resume restores the augmentation settings; changing the mode requires a new output directory.

```sh
.venv/bin/python -m algorithms.ppo \
  --resume checkpoints/ppo_cnn2x2_d4_seed0/last.pt --iterations 20000

.venv/bin/python play.py --agent ppo \
  --checkpoint pretrained/ppo_d4_cnn2x2.pt --auto
```

## Recorded result

Completed 20,000 iterations. The best validation checkpoint at iteration 18,900 averages 3,049.1 moves over 10 fixed-seed games (`1000000…1000009`), reaching 4096 in 80% and 8192 in 10%. This run reduces direction bias but retains some bias and performs worse in full games than the original PPO.

The bundled weights are inference-only. Per-game validation results and log provenance are in [results.json](../assets/results.json).

![PPO · CNN2×2 · D4 augmentation](../assets/ppo_d4_cnn2x2.png)
