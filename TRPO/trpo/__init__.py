"""
TRPO (Trust Region Policy Optimization) 实现

基于论文: "Trust Region Policy Optimization" (Schulman et al., 2015)
https://arxiv.org/abs/1502.05477

优化版本新增:
- 多环境并行采样 (VecEnv)
- 奖励归一化
- 熵系数衰减
"""

from .agent import TRPOAgent
from .network import ActorNetwork, CriticNetwork
from .rollout_buffer import RolloutBuffer, RunningMeanStd
from .vec_env import SubprocVecEnv, DummyVecEnv, FrameStackVecEnv, make_vec_env

__all__ = [
    'TRPOAgent',
    'ActorNetwork',
    'CriticNetwork',
    'RolloutBuffer',
    'RunningMeanStd',
    'SubprocVecEnv',
    'DummyVecEnv',
    'FrameStackVecEnv',
    'make_vec_env',
]
