"""
游戏环境注册模块

管理所有可用的游戏环境类。
"""

from envs.games.video_pinball import VideoPinballEnv


# 游戏注册表：游戏名称 -> 环境类
GAME_REGISTRY = {
    'VideoPinball': VideoPinballEnv,
}


__all__ = ['GAME_REGISTRY']
