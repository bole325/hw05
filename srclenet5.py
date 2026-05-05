#!/usr/bin/env python3
"""
任务二：LeNet-5模型定义
参考：LeCun et al., 1998 "Gradient-based learning applied to document recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 网络结构
    输入: 28x28x1 (灰度图)
    输出: 10个类别概率
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # 第一层: 卷积 + 平均池化
        # 输入: 1x28x28 → 输出: 6x24x24 → 池化: 6x12x12
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 第二层: 卷积 + 平均池化
        # 输入: 6x12x12 → 输出: 16x8x8 → 池化: 16x4x4
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        # 16 * 4 * 4 = 256 → 120 → 84 → 10
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 卷积块1
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 卷积块2
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_info(self):
        """返回模型结构信息"""
        return {
            'name': 'LeNet-5',
            'params': self.count_parameters(),
            'layers': [
                ('Conv1', f'1→6, 5x5 → {self.conv1.out_channels}x24x24'),
                ('AvgPool1', f'6x12x12'),
                ('Conv2', f'6→16, 5x5 → {self.conv2.out_channels}x8x8'),
                ('AvgPool2', f'16x4x4'),
                ('FC1', f'256→120'),
                ('FC2', f'120→84'),
                ('FC3', f'84→10')
            ]
        }

# 测试代码
if __name__ == "__main__":
    model = LeNet5()
    print(model.get_info())
    
    # 测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\n输入shape: {dummy_input.shape}")
    print(f"输出shape: {output.shape}")
    print(f"参数量: {model.count_parameters():,}")