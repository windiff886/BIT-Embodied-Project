"""
DQN 智能体

封装 DQN 训练逻辑：
- epsilon-greedy 动作选择
- TD 目标计算
- 网络训练更新
- epsilon 衰减策略
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .replay_buffer import ReplayBuffer, Transition


@dataclass
class DQNAgent:
    """
    DQN (2013) 智能体
    
    实现 epsilon-greedy 策略和 TD 学习。
    """
    
    # 环境参数
    n_actions: int
    
    # 超参数 
    replay_capacity: int = 300_000      # 经验回放容量
    batch_size: int = 128                   # 训练批大小
    gamma: float = 0.99                    # 折扣因子
    learning_rate: float = 0.00025         # RMSProp 学习率
    epsilon_start: float = 1.0             # 初始探索率
    epsilon_end: float = 0.1               # 最终探索率
    epsilon_decay_frames: int = 1_000_000  # epsilon 衰减帧数
    warmup_frames: int = 50_000            # 预热帧数（不训练）
    
    # 设备和优化
    device: str = 'cuda'
    use_compile: bool = False              # 是否使用 torch.compile 加速
    
    # 内部状态 (post_init 中初始化)
    q_network: QNetwork = field(init=False)
    optimizer: optim.Optimizer = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)
    frame_count: int = field(init=False, default=0)
    
    def __post_init__(self):
        """初始化网络、优化器和经验回放"""
        # 创建 Q-Network
        self.q_network = QNetwork(self.n_actions).to(self.device)
        
        # 使用 torch.compile 加速 (PyTorch 2.0+)
        if self.use_compile:
            self.q_network = torch.compile(self.q_network)  
        
        # RMSProp 优化器 (论文使用)
        self.optimizer = optim.RMSprop(
            self.q_network.parameters(),
            lr=self.learning_rate,
            alpha=0.95,      # 平滑常数
            eps=0.01         # 数值稳定性
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.replay_capacity)
        
        # 帧计数器
        self.frame_count = 0
    
    @property
    def epsilon(self) -> float:
        """
        计算当前 epsilon 值 (线性衰减)
        
        Returns:
            当前探索率
        """
        if self.frame_count >= self.epsilon_decay_frames:
            return self.epsilon_end
        
        # 线性插值
        progress = self.frame_count / self.epsilon_decay_frames
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        使用 epsilon-greedy 策略选择动作
        
        Args:
            state: 当前状态 (4, 84, 84)
            eval_mode: 是否为评估模式 (使用固定低 epsilon)
        
        Returns:
            选择的动作索引
        """
        # 评估模式使用固定低 epsilon
        eps = 0.10 if eval_mode else self.epsilon
        
        if random.random() < eps:
            # 随机探索
            return random.randint(0, self.n_actions - 1)
        else:
            # 利用：选择 Q 值最大的动作
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def select_actions_batch(self, states: np.ndarray, eval_mode: bool = False) -> List[int]:
        """
        批量选择动作（用于多环境并行）
        
        Args:
            states: 批量状态 (N, 4, 84, 84)
            eval_mode: 是否为评估模式
        
        Returns:
            动作列表，长度为 N
        """
        n_envs = states.shape[0]
        eps = 0.05 if eval_mode else self.epsilon
        
        # 批量计算 Q 值
        with torch.no_grad():
            states_tensor = torch.from_numpy(states).to(self.device)
            q_values = self.q_network(states_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        
        # 对每个环境应用 epsilon-greedy
        actions = []
        for i in range(n_envs):
            if random.random() < eps:
                actions.append(random.randint(0, self.n_actions - 1))
            else:
                actions.append(int(greedy_actions[i]))
        
        return actions
    
    def store_transition(self, transition: Transition):
        """
        存储经验到回放缓冲区
        
        Args:
            transition: 单步经验
        """
        self.replay_buffer.push(transition)
        self.frame_count += 1
    
    def train_step(self) -> Optional[float]:
        """
        执行一步训练更新
        
        Returns:
            训练损失值，若未训练则返回 None
        """
        # 预热期不训练
        if self.frame_count < self.warmup_frames:
            return None
        
        # 缓冲区数据不足
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样 mini-batch
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.from_numpy(
            np.stack([t.state for t in transitions])
        ).to(self.device)
        
        actions = torch.tensor(
            [t.action for t in transitions],
            dtype=torch.long,
            device=self.device
        )
        
        rewards = torch.tensor(
            [t.reward for t in transitions],
            dtype=torch.float32,
            device=self.device
        )
        
        next_states = torch.from_numpy(
            np.stack([t.next_state for t in transitions])
        ).to(self.device)
        
        dones = torch.tensor(
            [t.done for t in transitions],
            dtype=torch.float32,
            device=self.device
        )
        
        # 计算当前 Q 值: Q(s, a)
        current_q = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算 TD 目标: y = r + γ * max_a' Q(s', a')
        with torch.no_grad():
            next_q = self.q_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            # 终止状态的 TD 目标只有即时奖励
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 计算 MSE 损失
        loss = nn.functional.mse_loss(current_q, target_q)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # 别名方法，保持兼容
    def update(self) -> Optional[float]:
        """train_step 的别名"""
        return self.train_step()

    def get_epsilon(self) -> float:
        """获取当前 epsilon 值"""
        return self.epsilon

    def save(self, path: str):
        """
        保存模型权重
        
        Args:
            path: 保存路径
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_count': self.frame_count,
        }, path)
    
    def load(self, path: str):
        """
        加载模型权重
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame_count = checkpoint['frame_count']
