# 2048-rl-lab

[English](README.md) | **简体中文**

**在统一的 2048 环境中研究搜索与强化学习。**

2048-rl-lab 是一个研究代码库，用于分析搜索预算、策略优化和网络结构如何影响随机序贯决策问题中的算法表现。项目提供独立的算法实现、可复现的训练与评估流程、真实训练曲线，以及用于检查策略行为的预训练模型。

## 研究范围

| 方法 | 实现 | 模型与计算方式 |
| --- | --- | --- |
| [蒙特卡洛树搜索（MCTS）](algorithms/mcts.py) | UCT 选择、合法动作展开、随机模拟与回传 | 无神经网络；每步模拟次数可配置 |
| [优势 Actor–Critic（A2C）](algorithms/a2c.py) | 完整游戏采集、GAE、一次全批次更新 | MLP，133,381 个参数 |
| [近端策略优化（PPO）](algorithms/ppo.py) | GAE、随机打乱 minibatch、裁剪策略目标与 KL 检查 | MLP / ResCNN，133,381 / 175,877 个参数 |

项目支持研究策略学习与在线搜索的差异、MLP 与卷积表示的影响，以及存活步数、棋盘总量和大块达标率之间的关系。每个算法的训练循环保留在独立文件中；公共模块处理模型、轨迹、评估和 checkpoint 管理。

## 实验结果

下列图片从已有训练日志重新生成。左图展示每局平均步数；右图展示验证局达到 **512、1024、2048、4096** 的比例。阈值包含更大的块：达到 2048 的局也计入 512 和 1024。

| 方法 / 模型 | 训练迭代 | 选中模型迭代 | 验证局数 | 平均步数 | ≥2048 | ≥4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2C / MLP | 20,000 | 16,675 | 10 | 1,479.8 | 90% | 0% |
| PPO / MLP | 20,000 | 18,900 | 10 | 1,188.3 | 70% | 0% |
| PPO / ResCNN | 20,000 | 19,600 | 10 | 1,804.1 | 90% | 20% |

**评估设置：** 以上实验使用训练 seed 0，验证环境固定使用 `1000000…1000009`。模型按验证累计生成总量的均值选择。这些是**用于选择 checkpoint 的验证结果**，不是独立测试结果，也不是多个训练 seed 的平均结果，不能据此得出算法或网络结构的一般性排名。模型元数据和逐局记录见 [results.json](assets/results.json)。

### A2C · MLP

![A2C MLP：存活步数与大块达标率](assets/a2c_mlp.png)

### PPO · MLP

![PPO MLP：存活步数与大块达标率](assets/ppo_mlp.png)

### PPO · ResCNN

![PPO ResCNN：存活步数与大块达标率](assets/ppo_rescnn.png)

横轴是训练迭代，不是游戏局数，也不是单次优化器更新。存活步数图包含训练原始值、最近 50 次迭代均值、固定 seed 验证值，以及最近 5 次验证均值。星号标记验证回报最高的模型，不一定对应平均存活步数最高的位置。MCTS 没有训练曲线，应在明确的搜索预算下评估其表现。

## 环境与学习目标

[Gymnasium 环境](gym2048_env.py) 使用 4×4 棋盘和四个方向动作，并屏蔽非法动作。初始棋盘生成**一个块**；每次合法移动后，以 0.9 的概率生成 2，以 0.1 的概率生成 4。达到 2048 后继续运行，直到没有合法动作。

当前学习目标是**累计生成的新块总量**，鼓励更长的存活时间和更大的最终棋盘总量。训练内部将奖励除以 128，默认使用 `gamma=1` 和 GAE `lambda=0.95`。日志中的回报使用未缩放的原始值。

| 指标 | 定义 |
| --- | --- |
| `steps` / `mean_steps` | 单局步数 / 平均步数 |
| `spawn_return` / `mean_return` | 新生成块的累计总量 / 其逐局平均值 |
| `board_sum` | 最终棋盘所有数字之和，等于初始总量加上累计生成总量 |
| `merge_score` | 标准 2048 得分：所有合并事件中，合并后块的值之和 |
| `p512`、`p1024`、`p2048`、`p4096` | 达到至少相应块的游戏比例 |

旧日志中的 `mean_score` 是 `mean_return` 的别名，**不是**标准合并分数。与使用标准 2048 分数的研究比较时，应使用 `merge_score`；与初始生成两个块的实验比较前，也需要统一初始条件。

## 安装

需要 **Python 3.10+**。在仓库根目录运行：

```sh
git clone https://github.com/differentialmanifold/2048-rl-lab.git
cd 2048-rl-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

默认使用 CPU。具备相应 GPU 环境时，可指定 `--device cuda`。

## 训练

`--iterations` 必须指定，表示有限的**累计训练迭代目标**。每次从零开始的实验使用新的输出目录。

```sh
# A2C / MLP
.venv/bin/python -m algorithms.a2c \
  --architecture mlp --seed 0 --iterations 20000 \
  --episodes-per-update 8 --eval-every 25 --eval-episodes 10 \
  --plot-every 25 --save-dir checkpoints/a2c_mlp

# PPO / MLP
.venv/bin/python -m algorithms.ppo \
  --architecture mlp --seed 0 --iterations 20000 \
  --episodes-per-update 8 --epochs 4 --batch-size 256 \
  --eval-every 25 --eval-episodes 10 --plot-every 25 \
  --save-dir checkpoints/ppo_mlp

# PPO / ResCNN
.venv/bin/python -m algorithms.ppo \
  --architecture rescnn --seed 0 --iterations 20000 \
  --episodes-per-update 8 --epochs 4 --batch-size 256 \
  --eval-every 25 --eval-episodes 10 --plot-every 25 \
  --save-dir checkpoints/ppo_rescnn
```

每次迭代先采集八局完整游戏，再更新参数。A2C 对整批数据进行一次优化器更新；PPO 对随机打乱的 minibatch 最多训练四遍，默认 KL 阈值 `0.02` 可以提前结束当前批次的优化。每 25 次迭代及最后一次迭代后进行验证，使用更新后的策略，选择概率最大的合法动作。

训练环境的逐局 seed 为 `10000000 + seed + iteration × 100000 + episode_index`。固定实验 seed 用于在相同执行设置下复现训练，不表示每局重复同一棋盘。

### 继续实验

```sh
.venv/bin/python -m algorithms.ppo \
  --resume checkpoints/ppo_rescnn/last.pt --iterations 30000 --plot-every 25
```

如果已经完成 20,000 次迭代，该命令会再训练 10,000 次。模型结构和未指定的超参数自动继承。已有 v3 checkpoint，例如 `checkpoints/ppo_rescnn_seed0/last.pt` 和 `checkpoints/a2c_2048_v3/last.pt`，仍可由对应算法的训练入口加载。若从旧 checkpoint 分支实验，增加新的 `--save-dir`。同一训练目录只应由一个进程写入。

### Checkpoint 与绘图

| 输出 | 用途 |
| --- | --- |
| `metrics.jsonl` | 每个完成的迭代一条记录，按间隔附加验证结果 |
| `last.pt` | 最近的模型、优化器和随机数状态，用于续训 |
| `best.pt` | 平均验证回报最高的模型 |
| `training.png` | 每 `--plot-every` 次迭代及最后一次迭代后覆盖 |

绘图和验证间隔相互独立。可以从已有日志重画，也可以更新上方展示的图片：

```sh
.venv/bin/python plot.py --logs checkpoints/a2c_2048_v3/metrics.jsonl --output assets/a2c_mlp.png --title 'A2C · MLP'
.venv/bin/python plot.py --logs checkpoints/ppo_2048_v3/metrics.jsonl --output assets/ppo_mlp.png --title 'PPO · MLP'
.venv/bin/python plot.py --logs checkpoints/ppo_rescnn_seed0/metrics.jsonl --output assets/ppo_rescnn.png --title 'PPO · ResCNN'
```

结果表格和 `assets/results.json` 对应导出模型时的快照，更换展示模型后应同步更新。导出精简推理模型：

```sh
.venv/bin/python -m common.checkpoints \
  --input checkpoints/ppo_rescnn/best.pt --output pretrained/ppo_rescnn.pt
```

随仓库提供的推理模型约 0.5–0.7 MB/个，不含优化器和随机数状态，不能用于续训。完整训练 checkpoint 在本地生成，不纳入 Git。

## 独立测试

使用独立的 seed 范围，进行完整游戏评估：

```sh
.venv/bin/python evaluate.py --agent ppo --checkpoint pretrained/ppo_rescnn.pt \
  --episodes 100 --seed 2000000 --output experiments/ppo_rescnn_test.json

.venv/bin/python evaluate.py --agent a2c --checkpoint pretrained/a2c_mlp.pt \
  --episodes 100 --seed 2000000 --output experiments/a2c_test.json

.venv/bin/python evaluate.py --agent mcts --budget 500 \
  --episodes 20 --seed 2000000 --output experiments/mcts_test.json
```

输出包含步数、棋盘总量、合并得分和大块达标率。搜索评估通常比纯策略推理耗时更长。进行比较研究时，应统一环境规则和测试 seed，并随结果报告搜索预算。

## 检查策略行为

终端接口用于定性检查策略决策。**每次启动默认从操作系统获取新的随机性**，用于环境和搜索，不需要指定固定 seed。

```sh
.venv/bin/python play.py --agent ppo --checkpoint pretrained/ppo_rescnn.pt
.venv/bin/python play.py --agent ppo --checkpoint pretrained/ppo_mlp.pt
.venv/bin/python play.py --agent a2c --checkpoint pretrained/a2c_mlp.pt
.venv/bin/python play.py --agent mcts --budget 500
```

直接按 Enter 执行算法的一步决策；`H` 查看建议；输入 `W/A/S/D` 后按 Enter 手动移动；`P` 自动继续；`Q` 退出。添加 `--auto --delay 0.05` 可观察完整轨迹。界面显示本次会话的 seed，需要复现特定轨迹时可显式传入 `--seed`。

## 模型与代码组织

两个模型均使用棋盘指数 embedding、合法动作屏蔽，以及策略和价值双头。[MLP](common/models.py) 包含两个 256 单元的隐藏层。ResCNN 在 embedding 外增加数值等级通道，经过 32 通道卷积和两个残差块，再将展平特征映射为 256 维。

| 位置 | 职责 |
| --- | --- |
| `algorithms/` | 独立的搜索与训练实现 |
| `common/models.py` | 网络结构和合法动作分布 |
| `common/rollout.py` | 完整游戏采集与 GAE 目标 |
| `common/evaluation.py` | 固定 seed 评估 |
| `common/training.py`、`common/checkpoints.py` | 实验配置、持久化、日志与定期绘图 |
| `board.py`、`gym2048_env.py` | 游戏规则和 Gymnasium 接口 |
| `evaluate.py`、`plot.py`、`play.py` | 评估、可视化与策略检查 |
| `assets/`、`pretrained/`、`checkpoints/` | 结果图片、推理权重与实验日志 |
| `tests/` | 算法、可复现性、checkpoint 和输出回归测试 |

```sh
.venv/bin/python -m pytest -q
```

## 开源协议

[MIT](LICENSE)。Copyright © 2026 differentialmanifold。
