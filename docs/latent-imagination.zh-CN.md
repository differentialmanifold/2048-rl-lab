# Latent Imagination RL · 隐空间想象强化学习

[English](latent-imagination.md) | **简体中文**

[实现](../algorithms/latent_imagination/)先构建神经世界模型，再让策略在模型产生的连续隐状态中想象对局、学习获得更长的存活时间。世界模型负责动作后的变化、随机落子、奖励、合法方向和终止预测；策略可选 ViT 或 CNN2×2。目前的策略优化器采用 PPO，候选状态采用 afterstate 表示。

想象对局只在重置时编码初始棋盘，后续完全由学得的动力学递推。棋盘规则与解码器提供离线监督和核验；策略训练时世界模型冻结，真实环境对局用于评价并选择 `best.pt`。

## 构建世界模型

```sh
.venv/bin/python -m algorithms.latent_imagination.pipeline world \
  --run-dir checkpoints/latent_world_seed0 \
  --device mps --workers 8 --seed 0 \
  --bootstrap-iterations 10000 \
  --episodes 64 --horizon 10 --trajectory-updates 2000 \
  --max-trajectory-updates 0 \
  --eval-every 200 --plot-every 25 --refresh-every 1000
```

`pipeline world` 自动完成三个阶段：

1. **基础拟合**：生成合成与边界棋盘数据，训练 tokenizer 和 1／3／10 步动力学。
2. **探索策略**：在基础模型内训练一个初版策略，产生长时间递推的隐空间轨迹。
3. **长轨迹拟合**：从探索轨迹补充监督，继续训练动力学、事件概率和状态读出；tokenizer 冻结，合法方向与事件先验共享概率。

长轨迹拟合是世界模型构建的正式阶段，由 pipeline 自动衔接。各阶段通过检查后推进；相同命令恢复进度。`--trajectory-updates` 是最低更新数，`--max-trajectory-updates 0` 不设上限。

目录包含 `base/`、`exploration/` 和最终的 `world/`，进度记录在 `world_pipeline.json`。若已有通过验收的基础模型，可指定 `--base-world <目录>`；若还有该模型上的策略，可加 `--policy-checkpoint <checkpoint>` 复用。构建完成后直接将整个 `--run-dir` 传给策略的 `--world-run`。

## 在隐空间中学习

### ViT · 动态学习率

```sh
.venv/bin/python -m algorithms.latent_imagination.pipeline imagine \
  --world-run checkpoints/latent_world_seed0 \
  --architecture vit \
  --save-dir checkpoints/latent_imagination_vit_seed0 \
  --iterations 20000 --device mps --workers 8 --seed 0 \
  --episodes-per-update 8 --epochs 4 --batch-size 256 \
  --gamma 0.999 --td-steps 10 --td-lambda 0.5 \
  --lr 3e-4 --lr-schedule adaptive_kl \
  --lr-min 1e-6 --lr-max 3e-4 --lr-patience 5 \
  --clip-range 0.2 --target-kl 0.02 \
  --entropy-coef 0.01 --value-coef 0.5 \
  --eval-every 25 --eval-episodes 20 --plot-every 25
```

`adaptive_kl` 连续 5 轮全轨迹 KL 超过目标的 1.5 倍，或 KL 提前停止且不足一遍更新时，学习率乘 0.8；连续 5 轮完成全部 epoch 且 KL 低于目标的一半时，乘 1.1。学习率限制在 `[1e-6, 3e-4]`。

### CNN2×2 · 固定学习率

```sh
.venv/bin/python -m algorithms.latent_imagination.pipeline imagine \
  --world-run checkpoints/latent_world_seed0 \
  --architecture cnn2x2 \
  --save-dir checkpoints/latent_imagination_cnn2x2_seed0 \
  --iterations 20000 --device mps --workers 8 --seed 0 \
  --episodes-per-update 8 --epochs 4 --batch-size 256 \
  --gamma 0.999 --td-steps 10 --td-lambda 0.5 \
  --lr 3e-4 --lr-schedule constant \
  --clip-range 0.2 --target-kl 0.02 \
  --entropy-coef 0.01 --value-coef 0.5 \
  --eval-every 25 --eval-episodes 20 --plot-every 25
```

两者使用相同的世界模型，每轮 8 局想象对局、最多 4 遍 minibatch 更新，TD(10, 0.5)，γ=0.999；每 25 轮在真实环境验证 20 局并绘图。两个实验的骨干和学习率调度都不同。

## 续训、评估与试玩

```sh
.venv/bin/python -m algorithms.latent_imagination.pipeline imagine \
  --resume checkpoints/latent_imagination_vit_seed0/last.pt \
  --iterations 20000 --device mps --workers 8

.venv/bin/python evaluate.py --agent latent_imagination \
  --checkpoint checkpoints/latent_imagination_vit_seed0/best.pt \
  --episodes 100 --seed 2000000 --workers 8 \
  --output experiments/latent_imagination_vit_test.json

.venv/bin/python play.py --agent latent_imagination \
  --checkpoint checkpoints/latent_imagination_vit_seed0/best.pt --auto
```

续训恢复世界模型、策略、优化器及学习率调度状态，CNN2×2 换成对应目录。旧 checkpoint 也可用新入口续训，传入原来的 `last.pt` 路径即可。

输出为 `last.pt`、`best.pt`、`metrics.jsonl`、`training.png` 和学习率曲线 `optimization.png`。每个训练目录保持一个写入进程。

## 当前结果

公开结果使用此前训练的基础模型和探索策略；从零命令的初始化与训练历史不同。

2026-10-11 快照：两种策略均已完成 20,000 轮。以下为真实环境的最佳 20 局固定 seed（`1000000…1000019`）验证结果，数据见 [latent_imagination_results.json](../assets/latent_imagination_results.json)。

| 策略 | 最佳轮次 | 平均步数 | Spawn return | ≥4096 | ≥8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT · adaptive KL | 18,925 | 5,731.3 | 12,614.8 | 95% | 80% |
| CNN2×2 · constant | 19,475 | 4,775.85 | 10,514.1 | 95% | 60% |

![Latent Imagination RL · ViT · adaptive KL](../assets/latent_imagination_vit.png)
