"""
Breakout 游戏环境

经典 Atari 2600 打砖块游戏，4 个离散动作。
"""

from typing import Optional
from envs.base_env import BaseGameEnv


class BreakoutEnv(BaseGameEnv):
    """
    Breakout 游戏环境
    
    游戏说明：
    - 控制屏幕底部的挡板左右移动
    - 使用球击碎屏幕顶部的砖块
    - 目标是击碎所有砖块获得高分
    - 5 条命，球落入底部即失去一条命
    
    动作空间（4 个动作）：
    - 0: NOOP（无操作）
    - 1: FIRE（发射球）
    - 2: RIGHT（向右移动）
    - 3: LEFT（向左移动）
    """
    
    # ALE ROM 标识符
    ROM_NAME = 'ALE/Breakout-v5'
    
    def __init__(self, frame_skip: int = 4, render_mode: Optional[str] = None):
        """
        初始化 Breakout 环境
        
        Args:
            frame_skip: 跳帧数量，默认 4
            render_mode: 渲染模式，'human' 或 None
        """
        super().__init__(
            rom_name=self.ROM_NAME,
            frame_skip=frame_skip,
            render_mode=render_mode
        )
    
    def get_game_name(self) -> str:
        """
        获取游戏名称
        
        Returns:
            'Breakout'
        """
        return 'Breakout'
