import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class MnistTest(nn.Module):
    # 初始化神经网络层
    def __init__(self):
        super().__init__()
        # # 展平
        # self.flatten = nn.Flatten()
        # # 使用模块封装多个层
        # self.linear_relu_stack = nn.Sequential(
        #     # 全连接
        #     nn.Linear(28*28, 512),
        #     # 线性激活
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     # 输出10个分类
        #     nn.Linear(512, 10),
        # )

        # 基于CSDN的算法
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )


    # 向前传播
    def forward(self, x):
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        # return logits

        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x


