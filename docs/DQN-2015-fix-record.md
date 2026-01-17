# DQN-2015 训练崩溃问题修复记录

**日期**: 2026-01-17

**问题**: 训练过程中出现"越训练效果越差"的现象

---

## 问题现象

从训练曲线 `results/photos/DQN-2015-version1.png` 观察到：

| 阶段 | Episode 范围 | 100-ep 平均奖励 | 趋势 |
|------|-------------|----------------|------|
| 上升期 | 0 - 15,000 | 1 → 25 | 快速学习 |
| 平台期 | 15,000 - 40,000 | 20-25 | 稳定 |
| 衰退期 | 40,000 - 82,000 | 20 → 10 | **持续恶化** |

最终平均奖励从峰值 ~25 下降到 ~10，性能下降约 60%。

---

## 根因分析

### 1. 缺少梯度裁剪（致命问题）

**原代码** (`dqn/agent.py:167-170`)：
```python
self.optimizer.zero_grad()
loss.backward()
# 没有梯度裁剪
self.optimizer.step()
```

**问题**：
- 偶尔出现的大梯度可以一次性破坏已学好的策略
- Loss 曲线中的 spike（高达 0.1）正是这种现象的体现
- 这是导致后期训练崩溃的最直接原因

### 2. 探索率衰减过快

**原配置**：
```python
epsilon_decay_frames: int = 1_000_000   # 100万帧衰减完
total_frames: 50_000_000                 # 总共5000万帧
```

**问题**：
- 仅用 2% 的训练时间完成探索率衰减
- 后 98% 的训练几乎全是 exploitation
- Replay Buffer 被大量相似的次优经验填满
- 缺乏多样性导致灾难性遗忘

### 3. RMSProp 非标准配置

**原配置**：
```python
self.optimizer = optim.RMSprop(
    self.q_network.parameters(),
    lr=self.learning_rate,
    alpha=0.95,
    momentum=0.95,  # 非标准
    eps=0.01
)
```

**问题**：
- 原始 DQN 论文使用不带 momentum 的 RMSProp
- `momentum=0.95` 引入额外惯性，可能导致优化过冲
- 与论文实现不一致

---

## 修复方案

### 修复 1：添加梯度裁剪

**文件**: `dqn/agent.py`

**修改内容**：

1. 添加超参数：
```python
grad_clip: float = 10.0  # 梯度裁剪阈值
```

2. 在 `train_step()` 中添加梯度裁剪：
```python
self.optimizer.zero_grad()
loss.backward()

# 梯度裁剪，防止梯度爆炸导致训练崩溃
nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)

self.optimizer.step()
```

### 修复 2：延长探索率衰减

**文件**: `dqn/agent.py`

**修改内容**：
```python
# 原配置
epsilon_decay_frames: int = 1_000_000

# 修改后（延长到总帧数的 20%）
epsilon_decay_frames: int = 10_000_000
```

**效果**：
- 前 1000 万帧（20%）：探索率 1.0 → 0.1
- 后 4000 万帧（80%）：探索率固定在 0.1
- 给智能体更充分的探索时间

### 修复 3：修正 RMSProp 配置

**文件**: `dqn/agent.py`

**修改内容**：
```python
# 原配置
self.optimizer = optim.RMSprop(
    self.q_network.parameters(),
    lr=self.learning_rate,
    alpha=0.95,
    momentum=0.95,  # 移除
    eps=0.01
)

# 修改后（论文原始配置）
self.optimizer = optim.RMSprop(
    self.q_network.parameters(),
    lr=self.learning_rate,
    alpha=0.95,
    eps=0.01
)
```

---

## 修复前后对比

| 配置项 | 修复前 | 修复后 |
|--------|--------|--------|
| 梯度裁剪 | 无 | `max_norm=10.0` |
| epsilon 衰减帧数 | 1,000,000 | 10,000,000 |
| RMSProp momentum | 0.95 | 无 |

---

## 预期效果

1. **梯度裁剪**：防止偶发的大梯度破坏策略，训练更稳定
2. **延长探索**：Replay Buffer 保持多样性，减少灾难性遗忘
3. **标准 RMSProp**：优化行为更可预测，与论文一致

---

## 修改的文件

- `DQN-2015/dqn/agent.py`
  - 第 41 行：`epsilon_decay_frames` 1M → 10M
  - 第 44 行：新增 `grad_clip: float = 10.0`
  - 第 71-77 行：移除 RMSProp 的 `momentum` 参数
  - 第 170-171 行：新增梯度裁剪代码

---

## 后续建议

重新训练后，如果仍有问题，可以考虑：

1. **进一步调整 epsilon 衰减**：可尝试 `epsilon_decay_frames = 20_000_000`
2. **降低学习率**：当使用大 batch size（256）时，可尝试 `lr = 0.0001`
3. **监控梯度范数**：记录每步的梯度范数，观察是否有异常