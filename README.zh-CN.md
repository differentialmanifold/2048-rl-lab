# 2048-rl-lab

[English](README.md) | **简体中文**

**在统一的 2048 环境中研究搜索与强化学习。**

项目提供独立的算法实现、已记录的学习曲线、可复现的训练流程和预训练模型，以更长的存活时间和更大的累计方块总量为目标。

## 算法

| 算法 | 核心流程 | 展示模型 |
| --- | --- | --- |
| [MCTS](algorithms/mcts_chance.py) | UCT 选择动作、重新采样随机落子、完整随机模拟 | 无神经网络 |
| [A2C](algorithms/a2c.py) | 采集新轨迹、计算 TD(λ) 优势、一次 actor–critic 更新 | CNN2×2 |
| [PPO](algorithms/ppo.py) | TD(λ) 优势、裁剪更新、随机 minibatch、KL 检查 | CNN2×2 / ViT |
| [AlphaZero](algorithms/alphazero.py) | 使用游戏规则搜索、拟合访问次数策略、D4 对称增强 | CNN2×2 |
| [MuZero](algorithms/muzero.py) | 学习表示与动力学、隐状态搜索、序列展开训练 | CNN2×2 |

## 已记录的结果

![各算法的存活步数和方块达标率](assets/overview.png)

| 算法／模型 | 每步搜索次数 | 平均步数 | ≥2048 | ≥4096 | ≥8192 | ≥16384 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MCTS | 100 | 1,103.6 | 30% | 0% | 0% | 0% |
| A2C · CNN2×2 † | 0 | 1,311.6 | 60% | 0% | 0% | 0% |
| PPO · CNN2×2 | 0 | 5,765.4 | 100% | 100% | 80% | 0% |
| PPO · ViT † | 0 | 1,165.5 | 60% | 0% | 0% | 0% |
| AlphaZero · CNN2×2 | 100 | 3,576.4 | 90% | 90% | 30% | 0% |
| MuZero · CNN2×2 | 100 | 800.5 | 0% | 0% | 0% | 0% |

所有行均使用 **10 局，环境 seed 为 `1000000…1000009`**。MCTS、AlphaZero 和 MuZero 每步搜索 100 次；A2C/PPO 直接使用策略网络。神经网络行对应随项目发布的 checkpoint 验证成绩，包含挑选 checkpoint 的影响。训练预算不同，这张图用于展示已记录的表现，不构成严格控制变量的排名。

**快照日期：2026-09-22。** A2C/PPO 日志包含 20,000 次迭代；AlphaZero、MuZero 仍在训练，分别截取至第 542、5,402 次迭代。† A2C/CNN2×2 和 PPO/ViT 产生于当前默认训练配置之前，作为历史结果保留，不能将差异仅归因于模型结构。checkpoint 迭代数、逐局成绩和来源信息见 [results.json](assets/results.json)，MCTS 逐局成绩见 [mcts.json](assets/mcts.json)。

下列图片根据公开的[日志快照](assets/logs)重画。左图展示训练与验证的平均步数；右图展示 **512 至 16384** 的累计达标率，出现更大方块时自动扩展。淡线表示原始数据，实线表示 50 次训练迭代或 5 次验证的滑动均值，星号标记最高验证 spawn 回报。某局达到大块，并不意味着选中的 checkpoint 能稳定达到它。MCTS 没有训练过程，因此在上方概览中展示。

### A2C · CNN2×2

![A2C · CNN2×2: survival and tile reach rates](assets/a2c_cnn2x2.png)

### PPO · CNN2×2

![PPO · CNN2×2: survival and tile reach rates](assets/ppo_cnn2x2.png)

### PPO · ViT

![PPO · ViT: survival and tile reach rates](assets/ppo_vit.png)

### AlphaZero · CNN2×2

![AlphaZero · CNN2×2: survival and tile reach rates](assets/alphazero_cnn2x2.png)

### MuZero · CNN2×2

![MuZero · CNN2×2: survival and tile reach rates](assets/muzero_cnn2x2.png)

## 安装

需要 **Python 3.10+**，在仓库根目录执行：

```sh
git clone https://github.com/differentialmanifold/2048-rl-lab.git
cd 2048-rl-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

训练依据 PyTorch 后端可用性自动选择 **CUDA → Apple MPS → CPU**，并根据 CPU 核数和游戏数量选择 worker 数量。采集和验证 worker 使用 CPU，父进程在所选设备上更新模型。可选覆盖参数：`--device cpu`、`--device mps`、`--device cuda:0`、`--workers 2`。

## 环境与训练目标

4×4 棋盘初始生成一个块。每次合法移动后，以 0.9 的概率生成 2，以 0.1 的概率生成 4；达到 2048 后继续游戏，直到无合法动作。策略会屏蔽非法动作。奖励固定采用 **spawn mass**，即每次实际生成的 2 或 4。合并不改变总量，因此终局棋盘总和等于初始总量加累计奖励。

所有神经网络算法默认使用 **TD(λ)，n=10、λ=0.5、γ=0.999**，价值标签混合一步至十步回报。A2C/PPO 使用采集时冻结的 critic 自举，AlphaZero/MuZero 使用保存的搜索价值。训练内部的奖励和价值除以 128，日志使用原始回报；真实终局自举为零。MuZero 还学习原始即时奖励，以及终局之后的零奖励序列。

## 从零训练

`--iterations` 指定有限的总迭代次数，每个新实验使用独立目录。以下五条命令使用当前默认配置，其中包括 `--td-steps 10 --td-lambda 0.5 --gamma 0.999`。

```sh
# A2C · CNN2×2
.venv/bin/python -m algorithms.a2c \
  --architecture cnn2x2 --seed 0 --iterations 20000 \
  --save-dir checkpoints/a2c_cnn2x2_td_seed0

# PPO · CNN2×2
.venv/bin/python -m algorithms.ppo \
  --architecture cnn2x2 --seed 0 --iterations 20000 \
  --save-dir checkpoints/ppo_cnn2x2_td_seed0

# PPO · ViT
.venv/bin/python -m algorithms.ppo \
  --architecture vit --seed 0 --iterations 20000 \
  --save-dir checkpoints/ppo_vit_td_seed0

# AlphaZero · CNN2×2
.venv/bin/python -m algorithms.alphazero \
  --architecture cnn2x2 --seed 0 --iterations 20000 \
  --save-dir checkpoints/alphazero_cnn2x2_td_seed0

# MuZero · CNN2×2
.venv/bin/python -m algorithms.muzero \
  --architecture cnn2x2 --seed 0 --iterations 20000 \
  --save-dir checkpoints/muzero_cnn2x2_td_seed0
```

| 算法 | 每次迭代局数 | 参数更新 | 验证／绘图间隔 |
| --- | ---: | --- | ---: |
| A2C | 8 | 一次整批更新 | 25 次迭代 |
| PPO | 8 | 最多四遍 minibatch；batch 256，KL 阈值 0.02 | 25 次迭代 |
| AlphaZero | 8 | 100 个 minibatch，每批 256；回放 20,000 个状态 | 10 次迭代 |
| MuZero | 8 | 100 个 minibatch，每批 64；模型展开 5 步；回放 128 局 | 10 次迭代 |

验证默认使用 10 个固定 seed 的游戏，训练结束时也会验证。搜索算法每步模拟 100 次。可通过 `--eval-every`、`--eval-episodes`、`--plot-every`、`--mcts-sims` 调整相应配置。训练 seed 0 标识实验，逐局 seed 随迭代改变，不会重复同一个游戏。

### 续训

```sh
.venv/bin/python -m algorithms.ppo \
  --resume checkpoints/ppo_cnn2x2_td_seed0/last.pt --iterations 30000
```

如果已经完成 20,000 次迭代，这条命令继续训练 10,000 次。恢复模型结构、未显式指定的超参数、优化器和随机状态；搜索算法还恢复回放数据。续训要求模型与 TD 配置一致。改变验证口径时使用新的 `--save-dir`，以重新计算该目录的基线。同一输出目录保持一个写入进程。

| 输出 | 用途 |
| --- | --- |
| `metrics.jsonl` | 已完成的训练迭代及定期验证结果 |
| `last.pt` | 可续训的完整 checkpoint |
| `best.pt` | 当前验证口径下平均 spawn 回报最高的模型 |
| `training.png` | 运行中定期覆盖，结束时再次更新 |

```sh
.venv/bin/python plot.py --logs checkpoints/ppo_cnn2x2_td_seed0/metrics.jsonl
.venv/bin/python plot.py --logs assets/logs/ppo_cnn2x2.jsonl \
  --output assets/ppo_cnn2x2.png --title 'PPO · CNN2×2'
.venv/bin/python plot.py --results assets/results.json --output assets/overview.png
```

## 体验预训练模型

各神经网络算法默认载入项目附带的 CNN2×2 模型，PPO 另提供 ViT。交互游戏**每次启动使用新的随机性**，只有需要复现时才显式传入 `--seed`。

```sh
.venv/bin/python play.py --agent mcts --budget 100
.venv/bin/python play.py --agent a2c
.venv/bin/python play.py --agent ppo
.venv/bin/python play.py --agent ppo --checkpoint pretrained/ppo_vit.pt
.venv/bin/python play.py --agent alphazero --budget 100
.venv/bin/python play.py --agent muzero --budget 100
```

按 Enter 执行一步，`H` 获取建议，`W/A/S/D` 手动移动，`P` 自动继续，`Q` 退出。添加 `--auto --delay 0.05` 可观察整局。体验自己训练的模型时，传入 `--checkpoint checkpoints/<run>/best.pt`。

附带权重仅供推理，不包含续训所需状态。导出新模型：

```sh
.venv/bin/python -m common.checkpoints \
  --input checkpoints/ppo_cnn2x2_td_seed0/best.pt --output pretrained/ppo_cnn2x2.pt
```

## 评估

测试使用独立的 seed 范围：

```sh
.venv/bin/python evaluate.py --agent ppo --checkpoint pretrained/ppo_cnn2x2.pt \
  --episodes 100 --seed 2000000 --output experiments/ppo_cnn2x2.json

.venv/bin/python evaluate.py --agent mcts --budget 100 \
  --episodes 10 --seed 2000000 --output experiments/mcts.json

.venv/bin/python evaluate.py --agent alphazero --budget 100 \
  --episodes 10 --seed 2000000 --output experiments/alphazero.json

.venv/bin/python evaluate.py --agent muzero --budget 100 \
  --episodes 10 --seed 2000000 --output experiments/muzero.json
```

输出平均步数、累计生成总量、终局棋盘总和及各级方块的累计达标率，并记录搜索预算。独立评估默认使用 CPU，可指定 `--device auto`；多个 worker 仍使用 CPU。

## 模型与目录

两个编码器都使用方块指数 embedding，输出 256 维特征。CNN2×2 包含两层无 padding 卷积，空间尺寸 **4×4 → 3×3 → 2×2**，通道 **64 → 128**，使用投影残差、GroupNorm 和 SiLU。ViT 使用 16 个格子 token、宽度 96、两层四头 Transformer、二维轴向 RoPE，并按格子顺序读出。

| 模型 | Actor–critic 参数量 | MuZero 参数量 |
| --- | ---: | ---: |
| CNN2×2 | 173,893 | 240,326 |
| ViT | 187,085 | 253,518 |

| 位置 | 职责 |
| --- | --- |
| `algorithms/` | 独立的 MCTS、A2C、PPO、AlphaZero、MuZero 流程 |
| `common/models.py`、`common/targets.py`、`common/rollout.py` | 编码器、TD 目标和批量采集 |
| `common/evaluation.py`、`common/parallel.py` | 评估和持久化 CPU worker |
| `common/training.py`、`common/checkpoints.py` | 配置、续训、checkpoint 和日志 |
| `board.py`、`gym2048_env.py` | 游戏规则和 Gymnasium 接口 |
| `play.py`、`evaluate.py`、`plot.py` | 交互体验、评估和绘图 |
| `assets/`、`pretrained/` | 公开结果／日志快照和推理权重 |
| `checkpoints/`、`experiments/` | 本地训练产物，不进入 Git |

```sh
.venv/bin/python -m pytest -q
```

## 协议

[MIT](LICENSE)。Copyright © 2026 differentialmanifold。
