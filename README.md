# 具身智能大作业 — 强化学习算法实验

北京理工大学具身智能课程大作业，实现并对比 DQN (2013)、DQN (2015 Nature) 和 TRPO 三种强化学习算法在 Atari 游戏上的表现。

## 项目结构

```
.
├── DQN-2013/                # DQN 2013 实现
│   ├── dqn/
│   │   ├── network.py       # Q-Network (2 层卷积)
│   │   ├── agent.py         # DQN Agent (ε-greedy, RMSProp)
│   │   └── replay_buffer.py # 单帧存储经验回放
│   └── train.ipynb          # 训练脚本
├── DQN-2015/                # DQN 2015 (Nature) 实现
│   ├── dqn/
│   │   ├── network.py       # Q-Network (3 层卷积, 512 FC)
│   │   ├── agent.py         # DQN Agent + 目标网络 + Huber Loss
│   │   └── replay_buffer.py # 单帧存储经验回放
│   ├── train.ipynb          # 训练脚本
│   └── videos/              # 评估录像
├── TRPO/                    # TRPO 实现
│   ├── trpo/
│   │   ├── network.py       # Actor-Critic 网络
│   │   ├── agent.py         # TRPO Agent (共轭梯度, 线性搜索)
│   │   ├── rollout_buffer.py# GAE 轨迹缓冲区
│   │   └── vec_env.py       # 并行环境封装
│   ├── train.ipynb          # Version 1 (单环境)
│   └── train_optimized.py   # Version 2 (多环境优化)
├── envs/                    # 游戏环境
│   ├── base_env.py          # 基础环境类
│   ├── wrappers.py          # 帧预处理、帧堆叠、奖励裁剪
│   └── games/
│       ├── breakout.py      # Breakout (4 动作)
│       └── video_pinball.py # Video Pinball (9 动作)
├── results/photos/          # 训练曲线图
├── pdf/                     # LaTeX 报告
│   └── report.tex
├── docs/                    # 开发文档
└── requirements.txt
```

## 算法概览

| | DQN 2013 | DQN 2015 | TRPO |
|---|---|---|---|
| 方法 | 值函数 (off-policy) | 值函数 (off-policy) | 策略梯度 (on-policy) |
| 网络 | 2 层 CNN, 256 FC | 3 层 CNN, 512 FC | 3 层 CNN × 2 (Actor-Critic) |
| 关键特性 | 经验回放 | 目标网络, Huber Loss | 信赖域约束, GAE |
| Breakout 最佳成绩 | ~25 (avg) | ~35 (avg) | ~11 (崩溃前峰值) |

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：PyTorch >= 2.0、Gymnasium[atari]、ale-py、OpenCV、NumPy、Matplotlib。


## 训练

### DQN-2013 / DQN-2015

使用 Jupyter Notebook 训练：

```bash
cd DQN-2013  # 或 DQN-2015
jupyter notebook train.ipynb
```

训练参数在 notebook 开头的配置区域修改，关键参数包括并行环境数、batch size、回放缓冲区容量等。

### TRPO

```bash
cd TRPO

# Version 1: Jupyter Notebook (单环境)
jupyter notebook train.ipynb

# Version 2: 命令行 (多环境优化)
python train_optimized.py --n_envs 8 --total_steps 10000000
```

## 游戏环境

```python
from envs import make_env

env = make_env('Breakout')       # 或 'VideoPinball'
obs = env.reset()
obs, reward, done, info = env.step(0)
env.close()
```

添加新游戏：在 `envs/games/` 下继承 `BaseGameEnv`，在 `envs/__init__.py` 中注册。

## 实验结果

详细实验报告见 `pdf/report.pdf`，包含：

- **第一章**：DQN 2013/2015、TRPO 算法原理
- **第二章**：DQN 2013 实验（Video Pinball 基线、并行环境、batch size 对比、缓冲区容量对比）
- **第三章**：DQN 2015 实验（缓冲区扩展、梯度裁剪修复、ε 衰减调优）
- **第四章**：TRPO 实验（熵坍缩分析、on-policy 方法在视觉任务上的困难）
- **第五章**：三种算法对比总结

训练曲线图位于 `results/photos/`。

## 效果演示

训练完成后的 Breakout 游戏评估录像位于 `results/video/`：

- `breakout-DQN-2013.mp4` — DQN 2013 智能体
- `breakout-DQN-2015.mp4` — DQN 2015 智能体
- `TRPO.mp4` — TRPO 智能体
