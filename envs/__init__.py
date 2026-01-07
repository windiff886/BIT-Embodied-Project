"""
ALE 游戏环境模块

提供统一的 Atari 游戏环境接口，用于 DQN 算法训练和推理。
"""

import gymnasium as gym
import ale_py

# 注册 ALE 环境，确保 Gymnasium 能找到 ALE 命名空间
gym.register_envs(ale_py)

from envs.base_env import BaseGameEnv
from envs.wrappers import FrameStack, preprocess_frame, ClipRewardEnv
from envs.games import GAME_REGISTRY
from envs.games.video_pinball import VideoPinballEnv


def make_env(game_name: str, frame_stack: int = 4, frame_skip: int = 4, 
             render_mode: str = None) -> BaseGameEnv:
    """
    工厂函数：创建指定游戏的环境实例
    
    Args:
        game_name: 游戏名称，如 'VideoPinball'
        frame_stack: 帧堆叠数量，默认 4
        frame_skip: 跳帧数量，默认 4
        render_mode: 渲染模式，'human' 或 None
    
    Returns:
        配置好的游戏环境实例
    
    Raises:
        ValueError: 不支持的游戏名称
    
    Example:
        >>> env = make_env('VideoPinball')
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(0)
    """
    if game_name not in GAME_REGISTRY:
        available = ', '.join(GAME_REGISTRY.keys())
        raise ValueError(f"不支持的游戏: {game_name}。可用游戏: {available}")
    
    # 创建基础环境
    env_class = GAME_REGISTRY[game_name]
    env = env_class(frame_skip=frame_skip, render_mode=render_mode)
    
    # 应用帧堆叠包装器
    if frame_stack > 1:
        env = FrameStack(env, num_stack=frame_stack)
    
    return env


__all__ = ['make_env', 'BaseGameEnv', 'FrameStack', 'preprocess_frame', 'ClipRewardEnv', 'VideoPinballEnv']
