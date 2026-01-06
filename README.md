# DQN 复现项目 - ALE 游戏环境

## 项目依赖

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from envs import make_env

# 创建 Video Pinball 环境
env = make_env('VideoPinball')

# 重置环境
obs = env.reset()

# 执行动作
obs, reward, done, info = env.step(0)

# 释放资源
env.close()
```

## 支持的游戏

- Video Pinball

## 添加新游戏

1. 在 `envs/games/` 下创建新文件
2. 继承 `BaseGameEnv` 类
3. 在 `envs/__init__.py` 中注册
