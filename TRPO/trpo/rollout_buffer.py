"""
Rollout Buffer (轨迹缓冲区)

TRPO 是 on-policy 算法，需要收集完整的轨迹数据进行策略更新。
与 DQN 的 Replay Buffer 不同，Rollout Buffer 在每次更新后清空。

存储内容:
- 状态 (states)
- 动作 (actions)
- 奖励 (rewards)
- 价值估计 (values)
- log 概率 (log_probs)
- 优势函数 (advantages)
- 回报 (returns)

优化版本支持:
- 多环境并行采样
- 奖励归一化 (Running Mean/Std)
"""

import numpy as np
import torch
from typing import Generator, Tuple, Optional


class RunningMeanStd:
    """
    运行时均值和标准差计算器 (Welford's online algorithm)

    用于奖励归一化，稳定训练过程。
    """

    def __init__(self, shape: Tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """使用增量更新公式"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化输入"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RolloutBuffer:
    """
    轨迹缓冲区

    收集一个 epoch 的经验数据，用于 TRPO 策略更新。
    支持 GAE (Generalized Advantage Estimation) 计算优势函数。
    """

    def __init__(
        self,
        buffer_size: int,
        state_shape: Tuple[int, ...] = (4, 84, 84),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu',
        n_envs: int = 1,
        normalize_rewards: bool = True
    ):
        """
        初始化缓冲区

        Args:
            buffer_size: 缓冲区大小 (每个环境的步数)
            state_shape: 状态形状
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            device: 计算设备
            n_envs: 并行环境数量
            normalize_rewards: 是否归一化奖励
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.n_envs = n_envs
        self.normalize_rewards = normalize_rewards

        # 总容量 = 每环境步数 * 环境数量
        self.total_size = buffer_size * n_envs

        self.ptr = 0
        self.path_start_idx = np.zeros(n_envs, dtype=np.int64)
        self.full = False

        # 预分配内存 (展平存储)
        self.states = np.zeros((self.total_size, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(self.total_size, dtype=np.int64)
        self.rewards = np.zeros(self.total_size, dtype=np.float32)
        self.values = np.zeros(self.total_size, dtype=np.float32)
        self.log_probs = np.zeros(self.total_size, dtype=np.float32)
        self.dones = np.zeros(self.total_size, dtype=np.float32)

        # 计算后填充
        self.advantages = np.zeros(self.total_size, dtype=np.float32)
        self.returns = np.zeros(self.total_size, dtype=np.float32)

        # 奖励归一化
        if normalize_rewards:
            self.reward_rms = RunningMeanStd()
            self.ret = np.zeros(n_envs)  # 用于计算折扣回报的累积器
        else:
            self.reward_rms = None

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        添加一步经验 (单环境版本，向后兼容)

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            value: V(s) 价值估计
            log_prob: 动作的 log 概率
            done: 是否终止
        """
        assert self.ptr < self.total_size, "Buffer overflow"

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray
    ):
        """
        批量添加经验 (多环境版本)

        Args:
            states: (n_envs, *state_shape)
            actions: (n_envs,)
            rewards: (n_envs,)
            values: (n_envs,)
            log_probs: (n_envs,)
            dones: (n_envs,)
        """
        n = len(states)
        assert self.ptr + n <= self.total_size, "Buffer overflow"

        # 奖励归一化
        if self.normalize_rewards and self.reward_rms is not None:
            # 更新折扣回报累积器
            self.ret = self.ret * self.gamma + rewards
            self.reward_rms.update(self.ret.reshape(-1))
            # 归一化奖励
            rewards = rewards / np.sqrt(self.reward_rms.var + 1e-8)
            # 重置已结束环境的累积器
            self.ret[dones] = 0

        self.states[self.ptr:self.ptr + n] = states
        self.actions[self.ptr:self.ptr + n] = actions
        self.rewards[self.ptr:self.ptr + n] = rewards
        self.values[self.ptr:self.ptr + n] = values
        self.log_probs[self.ptr:self.ptr + n] = log_probs
        self.dones[self.ptr:self.ptr + n] = dones.astype(np.float32)

        self.ptr += n

    def finish_path(self, last_value: float = 0.0):
        """
        完成一条轨迹，计算 GAE 优势函数和回报 (单环境版本)

        当一个 episode 结束或缓冲区满时调用。

        Args:
            last_value: 最后状态的价值估计 (如果 episode 未结束)
                       如果 episode 已结束，则为 0
        """
        path_slice = slice(self.path_start_idx[0] if self.n_envs == 1 else 0, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        # 添加 bootstrap value
        values_extended = np.append(values, last_value)

        # GAE 计算
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        # A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        path_length = len(rewards)
        advantages = np.zeros(path_length, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(path_length)):
            if t == path_length - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        self.advantages[path_slice] = advantages
        # 回报 = 优势 + 价值估计
        self.returns[path_slice] = advantages + values

        if self.n_envs == 1:
            self.path_start_idx[0] = self.ptr

    def compute_gae_batch(self, last_values: np.ndarray):
        """
        批量计算 GAE (多环境版本)

        Args:
            last_values: (n_envs,) 每个环境最后状态的价值估计
        """
        # 重塑为 (steps_per_env, n_envs) 格式
        steps_per_env = self.ptr // self.n_envs

        rewards = self.rewards[:self.ptr].reshape(steps_per_env, self.n_envs)
        values = self.values[:self.ptr].reshape(steps_per_env, self.n_envs)
        dones = self.dones[:self.ptr].reshape(steps_per_env, self.n_envs)

        advantages = np.zeros_like(rewards)
        gae = np.zeros(self.n_envs)

        for t in reversed(range(steps_per_env)):
            if t == steps_per_env - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        self.advantages[:self.ptr] = advantages.flatten()
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        获取缓冲区中的所有数据

        Returns:
            states, actions, old_log_probs, advantages, returns 的元组
        """
        # 优势标准化 (减少方差)
        advantages = self.advantages[:self.ptr].copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data = (
            torch.from_numpy(self.states[:self.ptr]).to(self.device),
            torch.from_numpy(self.actions[:self.ptr]).to(self.device),
            torch.from_numpy(self.log_probs[:self.ptr]).to(self.device),
            torch.from_numpy(advantages).float().to(self.device),
            torch.from_numpy(self.returns[:self.ptr]).to(self.device),
        )

        return data

    def sample_batch(
        self,
        batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        随机批次采样生成器

        Args:
            batch_size: 批次大小

        Yields:
            每个批次的数据元组
        """
        indices = np.random.permutation(self.ptr)

        # 优势标准化
        advantages = self.advantages[:self.ptr].copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        start_idx = 0
        while start_idx < self.ptr:
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield (
                torch.from_numpy(self.states[batch_indices]).to(self.device),
                torch.from_numpy(self.actions[batch_indices]).to(self.device),
                torch.from_numpy(self.log_probs[batch_indices]).to(self.device),
                torch.from_numpy(advantages[batch_indices]).float().to(self.device),
                torch.from_numpy(self.returns[batch_indices]).to(self.device),
            )

            start_idx += batch_size

    def reset(self):
        """重置缓冲区"""
        self.ptr = 0
        self.path_start_idx = np.zeros(self.n_envs, dtype=np.int64)
        self.full = False

    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.ptr >= self.total_size

    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        return self.ptr
