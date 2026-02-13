# TRPO Version 1 vs Version 2 详细对比文档

本文档详细说明 TRPO 实现的两个版本之间的差异，包括算法原理、代码实现、超参数配置以及性能对比。

---

## 目录

1. [版本概述](#1-版本概述)
2. [核心问题分析：为什么 Version 1 表现差](#2-核心问题分析为什么-version-1-表现差)
3. [超参数对比](#3-超参数对比)
4. [架构改进](#4-架构改进)
5. [代码实现差异](#5-代码实现差异)
6. [理论分析](#6-理论分析)
7. [预期性能对比](#7-预期性能对比)
8. [使用指南](#8-使用指南)

---

## 1. 版本概述

### Version 1 (原始版本)

- **训练脚本**: `train.ipynb`
- **特点**: 基础 TRPO 实现，单环境采样
- **问题**: 训练效果差，奖励提升缓慢

### Version 2 (优化版本)

- **训练脚本**: `train_optimized.py`
- **特点**: 多环境并行、奖励归一化、探索增强
- **改进**: 全面优化样本效率和探索能力

---

## 2. 核心问题分析：为什么 Version 1 表现差

通过分析 `results/photos/TRPO-version1.png` 训练曲线，我们发现以下核心问题：

### 2.1 样本效率极低 (On-Policy 算法的固有缺陷)

```
问题: TRPO 是 on-policy 算法，每次策略更新后必须丢弃所有旧数据
```

**Version 1 的数据利用情况**:
- 每次更新只使用 2048 步数据
- 更新后立即丢弃所有数据
- 10M 步训练 ≈ 4800 次策略更新

**对比 DQN**:
- 100 万经验回放缓冲区
- 每个经验可被重复使用多次
- 10M 步训练 ≈ 250 万次网络更新

### 2.2 Trust Region 约束过于保守

```python
# Version 1 的保守设置
max_kl = 0.01        # KL 散度约束过紧
damping = 0.1        # 阻尼系数过大
backtrack_coeff = 0.8  # 回溯系数过高
```

**后果**:
- 策略每次只能做很小的更新
- 线性搜索经常失败，回退到旧策略
- 学习速度极慢

### 2.3 探索不足

```python
# Version 1
entropy_coef = 0.01  # 熵系数太小
```

**后果**:
- 策略过早收敛到局部最优
- 无法发现高奖励策略（如打穿砖块边缘）
- 只学会简单的接球动作

### 2.4 Critic 训练不充分

```python
# Version 1
critic_train_iters = 5   # 迭代次数少
critic_lr = 1e-3         # 学习率过高，可能不稳定
```

**后果**:
- 价值估计不准确
- GAE 优势计算误差大
- 策略梯度方向不正确

### 2.5 单环境采样效率低

```
Version 1: 1 个环境 × 2048 步 = 2048 样本/更新
```

**后果**:
- 数据收集速度慢
- 样本多样性不足
- 训练时间长

---

## 3. 超参数对比

### 3.1 完整超参数对比表

| 参数 | Version 1 | Version 2 | 变化 | 说明 |
|------|-----------|-----------|------|------|
| **折扣因子** |||||
| `gamma` | 0.99 | 0.99 | 不变 | 标准设置 |
| **GAE 参数** |||||
| `gae_lambda` | 0.95 | 0.98 | +0.03 | 更长期的优势估计 |
| **Trust Region** |||||
| `max_kl` | 0.01 | 0.02 | **×2** | 放宽约束，允许更大更新 |
| `damping` | 0.1 | 0.05 | **÷2** | 降低阻尼，更激进更新 |
| **线性搜索** |||||
| `cg_iters` | 10 | 15 | +5 | 更精确求解自然梯度 |
| `backtrack_iters` | 10 | 15 | +5 | 更多搜索机会 |
| `backtrack_coeff` | 0.8 | 0.6 | -0.2 | 更快找到有效步长 |
| **Critic** |||||
| `critic_lr` | 1e-3 | 5e-4 | ÷2 | 更稳定的学习 |
| `critic_train_iters` | 5 | 10 | ×2 | 更充分的训练 |
| **探索** |||||
| `entropy_coef` | 0.01 | **0.05** | **×5** | 大幅增加探索 |
| `entropy_decay` | 无 | 0.9995 | 新增 | 探索→利用过渡 |
| `min_entropy_coef` | 无 | 0.01 | 新增 | 保持最低探索 |
| **数据收集** |||||
| `rollout_steps` | 2048 | 4096 | ×2 | 每次收集更多数据 |
| `batch_size` | 64 | 128 | ×2 | 更大批次 |
| `n_envs` | 1 | **8** | **×8** | 并行环境 |
| **总样本/更新** | 2048 | **32768** | **×16** | 大幅提升 |

### 3.2 关键变化解释

#### 3.2.1 `max_kl`: 0.01 → 0.02

```
KL 散度约束定义了 "Trust Region" 的大小

max_kl = 0.01 (Version 1):
- 新旧策略非常接近
- 每次更新幅度很小
- 需要很多次更新才能学到好策略

max_kl = 0.02 (Version 2):
- 允许策略变化更大
- 每次更新更有效
- 但不会太大导致训练崩溃
```

#### 3.2.2 `entropy_coef`: 0.01 → 0.05

```
熵正则化公式: L_total = L_policy + entropy_coef × H(π)

entropy_coef = 0.01:
- 探索激励弱
- 策略快速收敛
- 容易陷入局部最优

entropy_coef = 0.05:
- 强探索激励
- 策略保持多样性
- 有机会发现更好的策略
```

#### 3.2.3 `gae_lambda`: 0.95 → 0.98

```
GAE 公式: A_t = Σ (γλ)^l × δ_{t+l}

λ = 0.95: 更多依赖短期优势
λ = 0.98: 考虑更长期的回报

对于 Breakout 这种需要长期规划的游戏，
更大的 λ 有助于学习延迟奖励的策略
```

---

## 4. 架构改进

### 4.1 多环境并行采样

#### Version 1: 单环境

```
┌─────────────┐
│   Agent     │
└──────┬──────┘
       │ action
       ▼
┌─────────────┐
│  Single Env │ ──→ 2048 samples/update
└─────────────┘
```

#### Version 2: 多环境并行

```
┌─────────────┐
│   Agent     │
└──────┬──────┘
       │ batch actions (n_envs)
       ▼
┌─────────────────────────────────────┐
│  Env 1  │  Env 2  │ ... │  Env 8   │
└─────────────────────────────────────┘
              │
              ▼
      8 × 4096 = 32768 samples/update
```

**新增文件**: `trpo/vec_env.py`

```python
class SubprocVecEnv:
    """多进程并行环境"""
    # 每个环境在独立进程中运行
    # 真正的并行，充分利用多核 CPU

class DummyVecEnv:
    """单进程向量化环境"""
    # 顺序执行，用于调试
    # 避免多进程的复杂性

class FrameStackVecEnv:
    """帧堆叠包装器"""
    # 自动管理每个环境的帧缓冲区
    # 输出 (n_envs, 4, 84, 84) 格式
```

### 4.2 奖励归一化

#### Version 1: 原始奖励

```python
# 直接使用环境奖励
reward = env.step(action)[1]  # 可能是 0, 1, 2, ...
```

**问题**:
- 奖励尺度不稳定
- 不同游戏阶段奖励差异大
- 影响价值函数学习

#### Version 2: 归一化奖励

**新增类**: `RunningMeanStd`

```python
class RunningMeanStd:
    """Welford's online algorithm"""

    def update(self, x):
        # 增量更新均值和方差
        ...

    def normalize(self, x):
        return (x - self.mean) / sqrt(self.var + 1e-8)
```

**实现位置**: `RolloutBuffer.add_batch()`

```python
if self.normalize_rewards:
    # 更新折扣回报的运行统计量
    self.ret = self.ret * self.gamma + rewards
    self.reward_rms.update(self.ret)

    # 归一化奖励
    rewards = rewards / sqrt(self.reward_rms.var + 1e-8)
```

### 4.3 熵系数衰减

#### Version 1: 固定熵系数

```python
entropy_coef = 0.01  # 始终不变
```

#### Version 2: 衰减熵系数

```python
# 初始化
current_entropy_coef = 0.05  # 初始高探索

# 每次更新后衰减
current_entropy_coef = max(
    min_entropy_coef,           # 最低 0.01
    current_entropy_coef * 0.9995  # 衰减
)
```

**衰减曲线**:

```
更新次数    熵系数
0          0.0500
1000       0.0303
2000       0.0184
3000       0.0111
4000       0.0100 (达到最小值)
```

**好处**:
- 训练初期: 高探索，发现好策略
- 训练后期: 低探索，精细优化策略

---

## 5. 代码实现差异

### 5.1 Agent 批量操作

#### Version 1: 单样本处理

```python
def select_action(self, state: np.ndarray, ...) -> Tuple[int, float, float]:
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
    action, log_prob = self.actor.get_action(state_tensor, ...)
    value = self.critic(state_tensor)
    return action.item(), log_prob.item(), value.item()
```

#### Version 2: 新增批量方法

```python
def select_action_batch(
    self,
    states: np.ndarray,  # (n_envs, 4, 84, 84)
    eval_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量选择动作"""
    with torch.no_grad():
        state_tensor = torch.from_numpy(states).to(self.device)
        actions, log_probs = self.actor.get_action(state_tensor, ...)
        values = self.critic(state_tensor).squeeze(-1)

    return (
        actions.cpu().numpy(),      # (n_envs,)
        log_probs.cpu().numpy(),    # (n_envs,)
        values.cpu().numpy()        # (n_envs,)
    )
```

### 5.2 Rollout Buffer 多环境支持

#### Version 1: 单环境

```python
def __init__(self, buffer_size, ...):
    self.states = np.zeros((buffer_size, *state_shape), ...)
    # ...

def add(self, state, action, reward, value, log_prob, done):
    self.states[self.ptr] = state
    # ...
    self.ptr += 1
```

#### Version 2: 多环境

```python
def __init__(self, buffer_size, n_envs=1, normalize_rewards=True, ...):
    self.n_envs = n_envs
    self.total_size = buffer_size * n_envs
    self.states = np.zeros((self.total_size, *state_shape), ...)

    # 奖励归一化
    if normalize_rewards:
        self.reward_rms = RunningMeanStd()
        self.ret = np.zeros(n_envs)

def add_batch(self, states, actions, rewards, values, log_probs, dones):
    """批量添加 n_envs 个样本"""
    n = len(states)

    # 奖励归一化
    if self.normalize_rewards:
        self.ret = self.ret * self.gamma + rewards
        self.reward_rms.update(self.ret)
        rewards = rewards / np.sqrt(self.reward_rms.var + 1e-8)
        self.ret[dones] = 0  # 重置已结束环境

    self.states[self.ptr:self.ptr + n] = states
    # ...
    self.ptr += n

def compute_gae_batch(self, last_values: np.ndarray):
    """批量计算 GAE"""
    steps_per_env = self.ptr // self.n_envs

    # 重塑为 (steps, n_envs) 格式
    rewards = self.rewards[:self.ptr].reshape(steps_per_env, self.n_envs)
    values = self.values[:self.ptr].reshape(steps_per_env, self.n_envs)
    dones = self.dones[:self.ptr].reshape(steps_per_env, self.n_envs)

    # 向量化 GAE 计算
    advantages = np.zeros_like(rewards)
    gae = np.zeros(self.n_envs)

    for t in reversed(range(steps_per_env)):
        # 同时计算所有环境的 GAE
        ...
```

### 5.3 训练循环对比

#### Version 1 (`train.ipynb`)

```python
while agent.total_steps < TOTAL_TIMESTEPS:
    # 单环境收集数据
    for _ in range(ROLLOUT_STEPS):
        action, log_prob, value = agent.select_action(state)
        next_frame, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, value, log_prob, done)
        # ...

    # 更新
    update_info = agent.update()
```

#### Version 2 (`train_optimized.py`)

```python
while total_steps < args.total_steps:
    rollout_buffer.reset()

    # 多环境并行收集
    for step in range(args.rollout_steps):
        # 批量选择动作
        actions, log_probs, values = agent.select_action_batch(states)

        # 所有环境同时步进
        next_states, rewards, dones, infos = env.step(actions)

        # 批量存储
        rollout_buffer.add_batch(states, actions, rewards, values, log_probs, dones)

        total_steps += args.n_envs  # 每步增加 n_envs

    # 批量计算 GAE
    rollout_buffer.compute_gae_batch(last_values)

    # 更新 (使用更大的数据集)
    agent.rollout_buffer = rollout_buffer
    update_info = agent.update()

    # 熵系数衰减
    current_entropy_coef *= args.entropy_decay
```

---

## 6. 理论分析

### 6.1 样本效率分析

#### 每次更新的有效样本量

| 版本 | 环境数 | 步数/环境 | 总样本 | 相对效率 |
|------|--------|-----------|--------|----------|
| V1 | 1 | 2048 | 2,048 | 1× |
| V2 | 8 | 4096 | 32,768 | **16×** |

#### 训练 10M 步的更新次数

| 版本 | 样本/更新 | 总更新次数 |
|------|-----------|------------|
| V1 | 2,048 | 4,883 |
| V2 | 32,768 | 305 |

**注意**: 虽然 V2 更新次数少，但每次更新的样本量大 16 倍，策略改进更充分。

### 6.2 探索-利用权衡

#### 熵对策略分布的影响

```
H(π) = -Σ π(a|s) log π(a|s)

高熵 (探索): π ≈ [0.25, 0.25, 0.25, 0.25]  # 均匀分布
低熵 (利用): π ≈ [0.01, 0.01, 0.97, 0.01]  # 集中分布
```

#### Version 1 vs Version 2 的探索能力

```
V1: entropy_coef = 0.01
    → 策略快速收敛
    → 可能错过好策略

V2: entropy_coef = 0.05 → 0.01 (衰减)
    → 训练初期广泛探索
    → 发现好策略后逐渐收敛
    → 最终精细优化
```

### 6.3 Trust Region 大小的影响

TRPO 优化目标:

```
max_θ  E[π_θ(a|s)/π_old(a|s) × A(s,a)]
s.t.   E[KL(π_old || π_θ)] ≤ δ
```

| δ (max_kl) | 效果 |
|------------|------|
| 0.001 | 极保守，几乎不更新 |
| 0.01 (V1) | 保守，更新缓慢 |
| **0.02 (V2)** | **适中，有效更新** |
| 0.1 | 激进，可能不稳定 |

### 6.4 GAE λ 的作用

```
λ = 0: A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error, 高偏差低方差)
λ = 1: A_t = Σ γ^k r_{t+k} - V(s_t)           (Monte Carlo, 低偏差高方差)
```

| λ | 特点 | 适用场景 |
|---|------|----------|
| 0.95 (V1) | 偏向短期 | 简单任务 |
| **0.98 (V2)** | **偏向长期** | **Atari 等复杂任务** |

---

## 7. 预期性能对比

### 7.1 训练效率

| 指标 | Version 1 | Version 2 | 提升 |
|------|-----------|-----------|------|
| 数据收集效率 | 1× | 8× | **8倍** |
| 每次更新样本量 | 2,048 | 32,768 | **16倍** |
| 策略更新幅度 | 保守 | 适中 | 更有效 |
| 探索能力 | 弱 | **强** | 显著提升 |

### 7.2 预期学习曲线

```
奖励
↑
│                                    ╭──── V2 优化版
│                               ╭───╯
│                          ╭───╯
│                     ╭───╯
│                ╭───╯
│           ╭───╯
│      ╭───╯        ╭─────────────── V1 原版 (缓慢提升)
│ ╭───╯        ╭───╯
│╯        ╭───╯
├────────╯
└─────────────────────────────────────────→ 训练步数
0        2M        4M        6M        8M       10M
```

### 7.3 关键里程碑预期

| 里程碑 | Version 1 | Version 2 |
|--------|-----------|-----------|
| 首次得分 | ~500K 步 | ~100K 步 |
| 平均 10 分 | ~3M 步 | ~1M 步 |
| 平均 50 分 | 可能无法达到 | ~5M 步 |
| 平均 100 分 | 无法达到 | ~8M 步 |

---

## 8. 使用指南

### 8.1 运行 Version 1

```bash
# 使用 Jupyter Notebook
jupyter notebook train.ipynb
```

### 8.2 运行 Version 2

```bash
cd /path/to/BIT-Embodied-Project/TRPO

# 默认配置 (8 环境，10M 步)
python train_optimized.py

# 自定义配置
python train_optimized.py \
    --n_envs 8 \
    --total_steps 10000000 \
    --entropy_coef 0.05 \
    --max_kl 0.02

# 查看所有参数
python train_optimized.py --help
```

### 8.3 重要参数调整建议

#### 如果训练不稳定

```bash
# 降低 KL 约束，增加阻尼
python train_optimized.py --max_kl 0.01 --damping 0.1
```

#### 如果探索不足

```bash
# 增加熵系数，降低衰减率
python train_optimized.py --entropy_coef 0.08 --entropy_decay 0.9999
```

#### 如果显存不足

```bash
# 减少环境数和批次大小
python train_optimized.py --n_envs 4 --batch_size 64 --rollout_steps 256
```

#### 如果 CPU 瓶颈

```bash
# 使用单进程环境 (在代码中修改)
# vec_env = DummyVecEnv(env_fns)  # 替代 SubprocVecEnv
```

---

## 总结

Version 2 通过以下改进解决了 Version 1 的核心问题：

1. **样本效率**: 8 环境并行 + 更大 rollout → **16× 样本/更新**
2. **探索能力**: 熵系数 0.01 → 0.05 + 衰减机制 → **更好的策略发现**
3. **更新效率**: max_kl 0.01 → 0.02 → **更有效的策略改进**
4. **训练稳定性**: 奖励归一化 + Critic 训练加强 → **更稳定的学习**

这些改进共同作用，预期将 TRPO 在 Breakout 上的表现提升至可用水平。

---

## 9. 训练塌陷问题修复记录 (2026-02-06)

基于现有训练结果图 (`training_curves.png` / `TRPO-version1.png`) 的排查，出现了典型的中后期塌陷现象：

- 前期 reward 有提升 (约 10~11)
- 中后期评估 reward 长时间为 0
- KL 曲线在后段大量出现 0（对应策略更新失效）

### 9.1 根因分析

1. **目标不一致**  
   在 `TRPO/trpo/agent.py` 中，策略梯度使用了 `policy_loss + entropy_coef * entropy`，但线性搜索验收使用的是仅 `policy_loss`。  
   这会导致“更新方向”和“验收标准”不一致，线性搜索更容易持续失败。

2. **失败信号被掩盖**  
   线性搜索失败时 KL 被写成 `0.0`，在曲线上会误判为“非常稳定”，实际上是“更新回滚/无有效更新”。

3. **缺少自适应稳定机制**  
   训练过程中没有根据连续失败自动收紧 trust region（如降低 `max_kl`、提升 `damping`）。

### 9.2 已实施修复

#### A. `TRPO/trpo/agent.py`

- 统一优化目标：线性搜索验收改为与梯度同目标（`policy_loss + entropy_bonus`）
- 新增训练诊断输出字段：
  - `line_search_success`
  - `line_search_iters`
  - `step_fraction`（失败时为 `0.0`）
- 线性搜索失败时 `kl_divergence` 记录为 `NaN`（不再伪装为 `0.0`）

#### B. `TRPO/train_optimized.py`

- 新增“连续线性搜索失败”自适应稳定器：
  - 连续失败达到阈值后，自动降低 `max_kl`
  - 同时自动提高 `damping`
- 日志新增监控项：
  - `LS_OK`（线性搜索成功率）
  - `StepFrac`（平均接受步长比例）
  - 当前 `max_kl` / `damping`
- KL 图新增自适应 `Max KL` 曲线（不是固定水平线）

### 9.3 新增可调参数

`train_optimized.py` 新增参数：

- `--ls_fail_threshold`：连续线性搜索失败阈值（默认 `20`）
- `--max_kl_decay`：触发自适应时 `max_kl` 衰减系数（默认 `0.8`）
- `--min_max_kl`：`max_kl` 下界（默认 `0.005`）
- `--damping_growth`：触发自适应时 `damping` 增长系数（默认 `1.2`）
- `--max_damping`：`damping` 上界（默认 `0.2`）

### 9.4 推荐运行方式（修复后）

```bash
cd /path/to/BIT-Embodied-Project/TRPO
python train_optimized.py \
  --total_steps 10000000 \
  --n_envs 8 \
  --max_kl 0.02 \
  --damping 0.05 \
  --ls_fail_threshold 20
```

重点观察：

- `LS_OK` 是否长期过低（如 < 30%）
- `StepFrac` 是否长期接近 0
- `KL` 是否大量 `NaN`（说明线性搜索持续失败）

---

*文档版本: 2.1*
*最后更新: 2026-02-06*
