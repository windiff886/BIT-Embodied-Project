"""
Q-Network 定义 (Nature DQN 2015)

实现 2015 Nature DQN 论文中描述的 CNN 结构：
- 输入: (batch, 4, 84, 84) - 4 帧堆叠的灰度图
- Conv1: 32 个 8x8 卷积核, stride=4, ReLU
- Conv2: 64 个 4x4 卷积核, stride=2, ReLU
- Conv3: 64 个 3x3 卷积核, stride=1, ReLU
- FC: 512 个神经元, ReLU
- 输出: n_actions 个 Q 值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Nature DQN (2015) Q-Network

    相比 2013 版本，网络更深更宽：
    - 增加第三层卷积 (64 filters, 3x3)
    - 全连接层从 256 增加到 512

    输入形状: (batch_size, 4, 84, 84)
    输出形状: (batch_size, n_actions)
    """

    def __init__(self, n_actions: int):
        super(QNetwork, self).__init__()

        # 卷积层
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
        # 计算卷积后的特征图大小:
        # 输入: 84x84
        # conv1 后: (84-8)/4 + 1 = 20 -> 20x20
        # conv2 后: (20-4)/2 + 1 = 9 -> 9x9
        # conv3 后: (9-3)/1 + 1 = 7 -> 7x7
        # 展平: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入状态，形状 (batch, 4, 84, 84)，值域 [0, 255] 或 [0, 1]

        Returns:
            各动作的 Q 值，形状 (batch, n_actions)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values


def create_q_network(n_actions: int, device: str = 'cpu') -> QNetwork:
    """
    创建 Q-Network 并移动到指定设备
    """
    network = QNetwork(n_actions)
    return network.to(device)
