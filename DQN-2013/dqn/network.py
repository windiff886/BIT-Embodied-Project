"""
Q-Network 定义

实现 2013 DQN 论文中描述的 CNN 结构：
- 输入: (batch, 4, 84, 84) - 4 帧堆叠的灰度图
- Conv1: 16 个 8x8 卷积核, stride=4, ReLU
- Conv2: 32 个 4x4 卷积核, stride=2, ReLU
- FC: 256 个神经元, ReLU
- 输出: n_actions 个 Q 值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    DQN (2013) Q-Network
    
    输入形状: (batch_size, 4, 84, 84)
    输出形状: (batch_size, n_actions)
    """
    
    def __init__(self, n_actions: int):
        """
        初始化 Q-Network
        
        Args:
            n_actions: 动作空间大小（输出 Q 值的数量）
        """
        super(QNetwork, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=4,      # 4 帧堆叠
            out_channels=16,    # 16 个滤波器
            kernel_size=8,      # 8x8 卷积核
            stride=4            # 步长 4
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,    # 32 个滤波器
            kernel_size=4,      # 4x4 卷积核
            stride=2            # 步长 2
        )
        
        # 全连接层
        # 计算卷积后的特征图大小:
        # 输入: 84x84
        # conv1 后: (84-8)/4 + 1 = 20 -> 20x20
        # conv2 后: (20-4)/2 + 1 = 9 -> 9x9
        # 展平: 32 * 9 * 9 = 2592
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态，形状 (batch, 4, 84, 84)，值域 [0, 255] 或 [0, 1]
        
        Returns:
            各动作的 Q 值，形状 (batch, n_actions)
        """
        # 如果输入是 uint8 [0, 255]，归一化到 [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # 卷积层 + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


def create_q_network(n_actions: int, device: str = 'cpu') -> QNetwork:
    """
    创建 Q-Network 并移动到指定设备
    
    Args:
        n_actions: 动作空间大小
        device: 计算设备 ('cpu' 或 'cuda')
    
    Returns:
        初始化后的 QNetwork 实例
    """
    network = QNetwork(n_actions)
    return network.to(device)
