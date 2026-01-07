"""
数据结构和经验回放

包含 Transition 和 ReplayBuffer。
"""

import random
from dataclasses import dataclass
from typing import List
from collections import deque
import numpy as np


@dataclass
class Transition:
    """
    单步经验数据

    Attributes:
        state: 当前状态 (4, 84, 84)
        action: 动作索引
        reward: 奖励 (-1/0/1)
        next_state: 下一状态 (4, 84, 84)
        done: 是否终止
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition: Transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
