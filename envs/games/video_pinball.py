"""
Video Pinball 游戏环境

Atari 2600 弹球游戏，9 个离散动作。
"""

from typing import Optional
from envs.base_env import BaseGameEnv


class VideoPinballEnv(BaseGameEnv):
    """
    Video Pinball 游戏环境
    
    游戏说明：
    - 控制弹球台的挡板和发射器
    - 目标是获得尽可能高的分数
    - 3 条命，球落入底部洞口即失去一条命
    
    动作空间（9 个动作）：
    - 0: NOOP（无操作）
    - 1: FIRE（发射）
    - 2: UP
    - 3: RIGHT
    - 4: LEFT
    - 5: DOWN
    - 6: UPFIRE
    - 7: RIGHTFIRE
    - 8: LEFTFIRE
    """
    
    # ALE ROM 标识符
    ROM_NAME = 'ALE/VideoPinball-v5'
    
    def __init__(self, frame_skip: int = 4, render_mode: Optional[str] = None):
        """
        初始化 Video Pinball 环境
        
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
            'VideoPinball'
        """
        return 'VideoPinball'
