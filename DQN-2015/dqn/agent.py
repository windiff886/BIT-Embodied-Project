"""
DQN 智能体 (Nature DQN 2015)

实现 2015 Nature 论文的关键改进：
- Target Network: 独立的目标网络计算 TD 目标
- 定期同步: 每隔固定步数将主网络参数复制到目标网络
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .replay_buffer import ReplayBuffer


def get_device() -> str:
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon
    else:
        return 'cpu'


@dataclass
class DQNAgent:
    """
    Nature DQN (2015) 智能体

    核心改进：使用 Target Network 稳定训练
    """

    # 环境参数
    n_actions: int
    num_envs: int = 1                      # 并行环境数量

    # 超参数
    replay_capacity: int = 1000_000        # 经验回放容量 (论文使用 1M)
    batch_size: int = 256                    # 训练批大小
    gamma: float = 0.99                     # 折扣因子
    learning_rate: float = 0.00025          # RMSProp 学习率
    epsilon_start: float = 1.0              # 初始探索率
    epsilon_end: float = 0.1                # 最终探索率
    epsilon_decay_frames: int = 10_000_000  # epsilon 衰减帧数 (延长到总帧数的20%)
    warmup_frames: int = 50_000             # 预热帧数
    target_update_freq: int = 10_000        # 目标网络更新频率 (关键参数)
    grad_clip: float = 10.0                 # 梯度裁剪阈值

    # 设备和优化
    device: str = 'cuda'
    use_compile: bool = False

    # 内部状态
    q_network: QNetwork = field(init=False)
    target_network: QNetwork = field(init=False)
    optimizer: optim.Optimizer = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)
    frame_count: int = field(init=False, default=0)

    def __post_init__(self):
        """初始化网络、目标网络、优化器和经验回放"""
        # 主 Q-Network
        self.q_network = QNetwork(self.n_actions).to(self.device)

        # Target Network (结构相同，参数独立)
        self.target_network = QNetwork(self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不需要梯度

        if self.use_compile:
            self.q_network = torch.compile(self.q_network)
            self.target_network = torch.compile(self.target_network)

        # RMSProp 优化器 (论文原始配置，不使用 momentum)
        self.optimizer = optim.RMSprop(
            self.q_network.parameters(),
            lr=self.learning_rate,
            alpha=0.95,
            eps=0.01
        )

        self.replay_buffer = ReplayBuffer(self.replay_capacity, num_envs=self.num_envs)
        self.frame_count = 0

    def sync_target_network(self):
        """将主网络参数同步到目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    @property
    def epsilon(self) -> float:
        """计算当前 epsilon 值 (线性衰减)"""
        if self.frame_count >= self.epsilon_decay_frames:
            return self.epsilon_end

        progress = self.frame_count / self.epsilon_decay_frames
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """使用 epsilon-greedy 策略选择动作"""
        eps = 0.05 if eval_mode else self.epsilon

        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def select_actions_batch(self, states: np.ndarray, eval_mode: bool = False) -> List[int]:
        """批量选择动作"""
        n_envs = states.shape[0]
        eps = 0.05 if eval_mode else self.epsilon

        with torch.no_grad():
            states_tensor = torch.from_numpy(states).to(self.device)
            q_values = self.q_network(states_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        actions = []
        for i in range(n_envs):
            if random.random() < eps:
                actions.append(random.randint(0, self.n_actions - 1))
            else:
                actions.append(int(greedy_actions[i]))

        return actions

    def store_transition(self, frame: np.ndarray, action: int, reward: float,
                         terminal: bool, env_id: int = 0):
        """存储单帧经验到回放缓冲区"""
        self.replay_buffer.push(frame, action, reward, terminal, env_id)

    def train_step(self) -> Optional[float]:
        """
        执行一步训练更新

        关键改进：使用 target_network 计算 TD 目标
        """
        if self.frame_count < self.warmup_frames:
            return None

        if not self.replay_buffer.can_sample(self.batch_size):
            return None

        # 新 API: sample() 返回元组 (states, actions, rewards, next_states, dones)
        states_np, actions_np, rewards_np, next_states_np, dones_np = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).long().to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        # 当前 Q 值: Q(s, a)
        current_q = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 使用 Target Network 计算 TD 目标 (2015 关键改进)
        # y = r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states)  # 使用目标网络
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Huber Loss (比 MSE 更鲁棒)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸导致训练崩溃
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)

        self.optimizer.step()

        # 定期同步目标网络
        if self.frame_count % self.target_update_freq == 0:
            self.sync_target_network()

        return loss.item()

    def update(self) -> Optional[float]:
        """train_step 的别名"""
        return self.train_step()

    def get_epsilon(self) -> float:
        """获取当前 epsilon 值"""
        return self.epsilon

    def save(self, path: str):
        """保存模型权重"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_count': self.frame_count,
        }, path)

    def load(self, path: str):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame_count = checkpoint['frame_count']
