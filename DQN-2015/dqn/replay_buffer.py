"""
经验回放缓冲区 (Experience Replay Buffer)

采用 DQN 2013 论文的内存优化机制：
- 只存储单帧，节省 75% 内存
- 采样时动态拼接 4 帧构建状态
- 正确处理 episode 边界
- 支持多环境：每个环境独立缓冲区，避免跨环境拼接
"""

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class _EnvBuffer:
    capacity: int
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    index: int = 0      # 下一个写入位置
    size: int = 0       # 当前有效数据量


class ReplayBuffer:
    """
    经验回放缓冲区（内存优化版）

    多环境场景下为每个环境维护独立缓冲区，采样时只在同一环境内拼帧。
    """

    def __init__(self, capacity: int, num_envs: int = 1,
                 frame_shape: Tuple[int, int] = (84, 84), history_len: int = 4):
        """
        Args:
            capacity: 总缓冲区容量（帧数）
            num_envs: 并行环境数量
            frame_shape: 单帧尺寸
            history_len: 堆叠帧数（构建状态用）
        """
        if num_envs <= 0:
            raise ValueError("num_envs 必须为正整数")
        if capacity < num_envs:
            raise ValueError("capacity 不能小于 num_envs")

        self.total_capacity = capacity
        self.num_envs = num_envs
        self.frame_shape = frame_shape
        self.history_len = history_len

        # 将总容量均分到各环境（余数分配给前几个环境）
        base = capacity // num_envs
        remainder = capacity % num_envs
        self.env_capacities = [
            base + (1 if i < remainder else 0) for i in range(num_envs)
        ]

        self.env_buffers = [self._init_env_buffer(c) for c in self.env_capacities]

    def _init_env_buffer(self, capacity: int) -> _EnvBuffer:
        frames = np.zeros((capacity, *self.frame_shape), dtype=np.uint8)
        actions = np.zeros(capacity, dtype=np.int32)
        rewards = np.zeros(capacity, dtype=np.float32)
        terminals = np.zeros(capacity, dtype=np.bool_)
        return _EnvBuffer(
            capacity=capacity,
            frames=frames,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )

    def _get_env_buffer(self, env_id: int) -> _EnvBuffer:
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id 超出范围: {env_id}")
        return self.env_buffers[env_id]

    def push(self, frame: np.ndarray, action: int, reward: float, terminal: bool,
             env_id: int = 0):
        """
        存储单帧经验

        Args:
            frame: 单帧观察 (84, 84)，uint8
            action: 动作索引
            reward: 奖励（已裁剪）
            terminal: 是否为终止帧
            env_id: 环境编号
        """
        env_buf = self._get_env_buffer(env_id)
        env_buf.frames[env_buf.index] = frame
        env_buf.actions[env_buf.index] = action
        env_buf.rewards[env_buf.index] = reward
        env_buf.terminals[env_buf.index] = terminal

        # 更新游标（循环）
        env_buf.index = (env_buf.index + 1) % env_buf.capacity
        env_buf.size = min(env_buf.size + 1, env_buf.capacity)

    def _get_state(self, env_buf: _EnvBuffer, index: int) -> np.ndarray:
        """
        从指定索引构建 4 帧堆叠状态

        处理两种边界情况：
        1. 缓冲区开头（数据不足）
        2. Episode 边界（不跨越 terminal）

        关键：一旦遇到 terminal，该位置及之前的所有帧都用零填充，
        避免混合不同 episode 的帧。
        """
        state = np.zeros((self.history_len, *self.frame_shape), dtype=np.uint8)

        # 从后往前扫描，找到最近的 terminal 边界
        # valid_start 表示从哪个位置开始取有效帧（之前的全部用零）
        valid_start = 0
        for i in range(self.history_len - 1, 0, -1):
            check_idx = (index - self.history_len + i) % env_buf.capacity
            if env_buf.terminals[check_idx]:
                valid_start = i
                break

        for i in range(valid_start, self.history_len):
            frame_idx = (index - self.history_len + 1 + i) % env_buf.capacity

            # 检查是否超出有效数据范围
            if env_buf.size < env_buf.capacity:
                if frame_idx >= env_buf.size or frame_idx > index:
                    continue

            state[i] = env_buf.frames[frame_idx]

        return state

    def _get_invalid_range(self, env_buf: _EnvBuffer) -> set:
        """
        返回不能采样的索引集合。

        无效索引的原因：
        1. buffer 未满时：前 history_len-1 个位置没有足够历史帧
        2. buffer 满时：index 附近的位置会导致 state 或 next_state 跨越循环断点，
           混合最新和最旧的数据。

        当 buffer 满时，无效范围是 {index-1, index, index+1, ..., index+history_len-2}，
        即从 index-1 开始的连续 history_len 个位置。
        """
        if env_buf.size < env_buf.capacity:
            return set(range(self.history_len - 1))
        # 返回 {index-1, index, index+1, ..., index+history_len-2}
        return set((env_buf.index - 1 + i) % env_buf.capacity for i in range(self.history_len))

    def _valid_count(self, env_buf: _EnvBuffer, invalid_range: set) -> int:
        max_index = env_buf.size - 1
        if max_index <= 0:
            return 0
        invalid_in_range = sum(1 for idx in invalid_range if idx < max_index)
        valid_count = max_index - invalid_in_range
        return max(0, valid_count)

    def can_sample(self, batch_size: int) -> bool:
        total_valid = 0
        for env_buf in self.env_buffers:
            invalid_range = self._get_invalid_range(env_buf)
            total_valid += self._valid_count(env_buf, invalid_range)
        return total_valid >= batch_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
        """
        随机采样 mini-batch（跨环境随机）

        Returns:
            (states, actions, rewards, next_states, dones)
            - states: (batch, 4, 84, 84) uint8
            - actions: (batch,) int32
            - rewards: (batch,) float32
            - next_states: (batch, 4, 84, 84) uint8
            - dones: (batch,) float32
        """
        valid_envs = []
        valid_counts = []
        invalid_ranges = []
        max_indices = []

        for env_id, env_buf in enumerate(self.env_buffers):
            invalid_range = self._get_invalid_range(env_buf)
            valid_count = self._valid_count(env_buf, invalid_range)
            if valid_count <= 0:
                continue
            valid_envs.append(env_id)
            valid_counts.append(valid_count)
            invalid_ranges.append(invalid_range)
            max_indices.append(env_buf.size - 1)

        total_valid = sum(valid_counts)
        if total_valid < batch_size:
            raise ValueError("可用样本不足，无法采样所需 batch_size")

        env_choices = random.choices(
            range(len(valid_envs)), weights=valid_counts, k=batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for choice in env_choices:
            env_id = valid_envs[choice]
            env_buf = self.env_buffers[env_id]
            invalid_range = invalid_ranges[choice]
            max_index = max_indices[choice]

            while True:
                idx = random.randrange(max_index)
                if idx not in invalid_range:
                    break

            states.append(self._get_state(env_buf, idx))
            actions.append(env_buf.actions[idx])
            rewards.append(env_buf.rewards[idx])
            next_states.append(self._get_state(env_buf, (idx + 1) % env_buf.capacity))
            dones.append(float(env_buf.terminals[idx]))

        return (
            np.stack(states),
            np.asarray(actions),
            np.asarray(rewards),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return sum(env_buf.size for env_buf in self.env_buffers)
