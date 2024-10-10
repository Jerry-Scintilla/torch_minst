import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class MnistTest(nn.Module):
    # 初始化神经网络层
    def __init__(self):
        super().__init__()
        # 展平
        self.flatten = nn.Flatten()
        # 使用模块封装多个层
        self.linear_relu_stack = nn.Sequential(
            # 全连接
            nn.Linear(28*28, 512),
            # 线性激活
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # 输出10个分类
            nn.Linear(512, 10),
        )

    # 向前传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits