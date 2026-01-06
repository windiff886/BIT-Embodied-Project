"""
帧处理工具和包装器

提供 DQN 所需的图像预处理和帧堆叠功能。
"""

from typing import Tuple
from collections import deque
import numpy as np
import cv2


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (84, 84)) -> np.ndarray:
    """
    预处理游戏帧，符合 DQN 论文规范
    
    处理步骤：
    1. 转换为灰度图
    2. 缩放至目标尺寸 (84x84)
    
    Args:
        frame: 原始 RGB 帧，形状 (H, W, 3)
        target_size: 目标尺寸，默认 (84, 84)
    
    Returns:
        预处理后的灰度帧，形状 (84, 84)，uint8 类型
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 缩放至目标尺寸
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    return resized.astype(np.uint8)


class FrameStack:
    """
    帧堆叠包装器
    
    将多个连续帧堆叠成单个观测，提供时序信息。
    符合 DQN 论文中的 4 帧堆叠设计。
    """
    
    def __init__(self, env, num_stack: int = 4):
        """
        初始化帧堆叠包装器
        
        Args:
            env: 底层游戏环境（BaseGameEnv 实例）
            num_stack: 堆叠帧数，默认 4
        """
        self._env = env
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
        
        # 更新观测形状
        base_shape = env.get_observation_shape()
        self._observation_shape = (num_stack, base_shape[0], base_shape[1])
    
    def reset(self, seed: int = None) -> np.ndarray:
        """
        重置环境并初始化帧栈
        
        Args:
            seed: 随机种子
        
        Returns:
            堆叠后的观测，形状 (num_stack, 84, 84)
        """
        obs = self._env.reset(seed=seed)
        
        # 用初始帧填充整个栈
        self._frames.clear()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        
        return self._get_stacked_obs()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作并更新帧栈
        
        Args:
            action: 动作索引
        
        Returns:
            stacked_obs: 堆叠后的观测 (num_stack, 84, 84)
            reward: 即时奖励
            done: 是否终止
            info: 额外信息
        """
        obs, reward, done, info = self._env.step(action)
        self._frames.append(obs)
        return self._get_stacked_obs(), reward, done, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """
        获取当前堆叠的观测
        
        Returns:
            堆叠后的帧，形状 (num_stack, 84, 84)
        """
        return np.stack(list(self._frames), axis=0)
    
    def get_action_space(self) -> int:
        """获取动作空间大小"""
        return self._env.get_action_space()
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """
        获取堆叠后的观测形状
        
        Returns:
            (num_stack, 84, 84)
        """
        return self._observation_shape
    
    def sample_action(self) -> int:
        """随机采样动作"""
        return self._env.sample_action()
    
    def render(self) -> None:
        """渲染当前帧"""
        self._env.render()
    
    def close(self) -> None:
        """释放资源"""
        self._env.close()
    
    def get_lives(self) -> int:
        """获取剩余生命数"""
        return self._env.get_lives()
    
    def get_game_name(self) -> str:
        """获取游戏名称"""
        return self._env.get_game_name()
    
    @property
    def unwrapped(self):
        """获取底层未包装的环境"""
        return self._env.unwrapped
