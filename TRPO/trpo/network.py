"""
TRPO 网络定义 (Actor-Critic 架构)

TRPO 使用分离的 Actor 和 Critic 网络:
- Actor: 输出动作的概率分布 (策略网络)
- Critic: 估计状态价值函数 V(s) (价值网络)

输入: (batch, 4, 84, 84) - 4 帧堆叠的灰度图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """
    策略网络 (Actor)

    输出离散动作的概率分布，使用与 DQN 相同的 CNN 骨干网络。

    输入形状: (batch_size, 4, 84, 84)
    输出形状: (batch_size, n_actions) - softmax 概率分布
    """

    def __init__(self, n_actions: int):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions

        # CNN 骨干网络 (与 Nature DQN 2015 相同)
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        # 全连接层
        # 卷积后: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # 初始化最后一层权重为较小值，使初始策略接近均匀分布
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回动作的 log 概率

        Args:
            x: 输入状态，形状 (batch, 4, 84, 84)

        Returns:
            动作的 log 概率，形状 (batch, n_actions)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # 使用 log_softmax 提高数值稳定性
        log_probs = F.log_softmax(self.fc2(x), dim=-1)

        return log_probs

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        """
        根据当前策略采样动作

        Args:
            x: 输入状态
            deterministic: 是否使用确定性策略 (选择概率最高的动作)

        Returns:
            action: 采样的动作
            log_prob: 该动作的 log 概率
        """
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()

        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """
        评估给定状态-动作对的 log 概率和熵

        Args:
            x: 状态批次
            actions: 动作批次

        Returns:
            log_probs: 动作的 log 概率
            entropy: 策略熵
        """
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)

        dist = Categorical(probs)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = dist.entropy()

        return action_log_probs, entropy


class CriticNetwork(nn.Module):
    """
    价值网络 (Critic)

    估计状态价值函数 V(s)，用于计算优势函数 A(s, a)。

    输入形状: (batch_size, 4, 84, 84)
    输出形状: (batch_size, 1) - 状态价值
    """

    def __init__(self):
        super(CriticNetwork, self).__init__()

        # CNN 骨干网络 (与 Actor 相同)
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

        # 初始化
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回状态价值

        Args:
            x: 输入状态，形状 (batch, 4, 84, 84)

        Returns:
            状态价值，形状 (batch, 1)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        value = self.fc2(x)

        return value


class ActorCriticNetwork(nn.Module):
    """
    共享骨干网络的 Actor-Critic 架构

    在某些情况下，共享 CNN 特征提取器可以提高效率。
    但 TRPO 论文中通常使用分离的网络。
    """

    def __init__(self, n_actions: int):
        super(ActorCriticNetwork, self).__init__()
        self.n_actions = n_actions

        # 共享 CNN 骨干
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 共享全连接层
        self.fc_shared = nn.Linear(64 * 7 * 7, 512)

        # Actor 头
        self.actor_head = nn.Linear(512, n_actions)

        # Critic 头
        self.critic_head = nn.Linear(512, 1)

        # 初始化
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Returns:
            log_probs: 动作 log 概率
            value: 状态价值
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

        log_probs = F.log_softmax(self.actor_head(x), dim=-1)
        value = self.critic_head(x)

        return log_probs, value
