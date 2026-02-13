"""
向量化环境 (Vectorized Environment)

通过并行运行多个环境实例，大幅提高 on-policy 算法的数据收集效率。
对于 TRPO 这种样本效率低的算法，这是关键优化。
"""

import numpy as np
from multiprocessing import Process, Pipe
from typing import List, Tuple, Callable, Optional
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def worker(remote, parent_remote, env_fn):
    """
    子进程 worker，运行单个环境实例
    """
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    # 自动重置
                    obs = env.reset()
                remote.send((obs, reward, done, info))

            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)

            elif cmd == 'close':
                remote.close()
                break

            elif cmd == 'get_spaces':
                remote.send((env.get_action_space(), env.get_observation_shape()))

        except EOFError:
            break


class SubprocVecEnv:
    """
    基于多进程的向量化环境

    每个环境运行在独立的子进程中，实现真正的并行。
    """

    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: 环境创建函数列表，每个函数返回一个环境实例
        """
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # 创建管道
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])

        # 启动子进程
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(target=worker, args=(work_remote, remote, env_fn))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()

        # 获取环境信息
        self.remotes[0].send(('get_spaces', None))
        self.n_actions, self.obs_shape = self.remotes[0].recv()

    def step_async(self, actions: np.ndarray):
        """异步执行动作"""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """等待所有环境完成步进"""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos)
        )

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """同步执行动作"""
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> np.ndarray:
        """重置所有环境"""
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self):
        """关闭所有环境"""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()


class DummyVecEnv:
    """
    简单的向量化环境（单进程，用于调试）

    按顺序运行多个环境，不使用多进程。
    """

    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.n_actions = self.envs[0].get_action_space()
        self.obs_shape = self.envs[0].get_observation_shape()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list
        )

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs])

    def close(self):
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()


class FrameStackVecEnv:
    """
    带帧堆叠功能的向量化环境包装器

    将连续 n_frames 帧堆叠为单个状态，提供时序信息。
    """

    def __init__(self, vec_env, n_frames: int = 4):
        self.vec_env = vec_env
        self.n_frames = n_frames
        self.n_envs = vec_env.n_envs
        self.n_actions = vec_env.n_actions

        # 帧缓冲区: (n_envs, n_frames, H, W)
        H, W = vec_env.obs_shape
        self.frames = np.zeros((self.n_envs, n_frames, H, W), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        """重置并填充初始帧"""
        obs = self.vec_env.reset()  # (n_envs, H, W)
        for i in range(self.n_frames):
            self.frames[:, i] = obs
        return self.frames.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """执行动作并更新帧堆叠"""
        obs, rewards, dones, infos = self.vec_env.step(actions)

        # 左移帧，添加新帧
        self.frames[:, :-1] = self.frames[:, 1:]
        self.frames[:, -1] = obs

        # 对于结束的环境，重置帧缓冲区
        for i, done in enumerate(dones):
            if done:
                for j in range(self.n_frames):
                    self.frames[i, j] = obs[i]

        return self.frames.copy(), rewards, dones, infos

    def close(self):
        self.vec_env.close()


def make_vec_env(env_fn: Callable, n_envs: int = 8, use_subproc: bool = True) -> FrameStackVecEnv:
    """
    创建向量化环境的便捷函数

    Args:
        env_fn: 环境创建函数
        n_envs: 并行环境数量
        use_subproc: 是否使用多进程

    Returns:
        带帧堆叠的向量化环境
    """
    env_fns = [env_fn for _ in range(n_envs)]

    if use_subproc:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return FrameStackVecEnv(vec_env)
