"""
游戏环境抽象基类

定义所有 ALE 游戏环境的统一接口。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
import gymnasium as gym

from envs.wrappers import preprocess_frame


class BaseGameEnv(ABC):
    """
    ALE 游戏环境的抽象基类
    
    所有具体游戏环境必须继承此类并实现抽象方法。
    提供 DQN 训练所需的标准接口。
    """
    
    def __init__(self, rom_name: str, frame_skip: int = 4, 
                 render_mode: Optional[str] = None):
        """
        初始化游戏环境
        
        Args:
            rom_name: ALE ROM 名称，如 'ALE/VideoPinball-v5'
            frame_skip: 跳帧数量，每个动作重复执行的帧数
            render_mode: 渲染模式，'human' 显示窗口，None 不渲染
        """
        self.rom_name = rom_name
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        
        # 创建底层 Gymnasium 环境
        self._env = gym.make(
            rom_name,
            frameskip=frame_skip,
            render_mode=render_mode
        )
        
        # 缓存环境信息
        self._action_space_size = self._env.action_space.n
        self._observation_shape = (84, 84)  # 预处理后的形状
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子，用于可重复性
        
        Returns:
            预处理后的初始观测帧 (84, 84)，uint8 类型
        """
        obs, info = self._env.reset(seed=seed)
        return preprocess_frame(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作并返回环境反馈
        
        Args:
            action: 动作索引，范围 [0, action_space_size)
        
        Returns:
            observation: 预处理后的观测帧 (84, 84)
            reward: 即时奖励
            done: 是否终止（包括 terminated 和 truncated）
            info: 额外信息字典
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return preprocess_frame(obs), reward, done, info
    
    def get_action_space(self) -> int:
        """
        获取动作空间大小
        
        Returns:
            可用动作的数量
        """
        return self._action_space_size
    
    def get_observation_shape(self) -> Tuple[int, int]:
        """
        获取单帧观测的形状
        
        Returns:
            观测帧形状 (84, 84)
        """
        return self._observation_shape
    
    def sample_action(self) -> int:
        """
        随机采样一个动作
        
        Returns:
            随机动作索引
        """
        return self._env.action_space.sample()
    
    def render(self) -> None:
        """
        渲染当前帧（仅在 render_mode='human' 时有效）
        """
        if self.render_mode == 'human':
            self._env.render()
    
    def close(self) -> None:
        """
        释放环境资源
        """
        self._env.close()
    
    @property
    def unwrapped(self):
        """获取底层未包装的环境"""
        return self._env.unwrapped
    
    def get_lives(self) -> int:
        """
        获取当前剩余生命数
        
        Returns:
            剩余生命数
        """
        return self._env.unwrapped.ale.lives()
    
    @abstractmethod
    def get_game_name(self) -> str:
        """
        获取游戏名称（子类必须实现）
        
        Returns:
            游戏名称字符串
        """
        pass
