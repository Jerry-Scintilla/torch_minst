from isexist import isexist
# 导入神经网络
from model import *

# 设置训练设备为cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载MNIST数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 加载数据集
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 实例化神经网络
if isexist(name='model.pth'):
    model = torch.load('model.pth')
else:
    model = MnistTest()
model = model.to(device)

# 设置训练参数
# 学习速率
learning_rate = 1e-3
# 包大小
batch_size = 64


# 设置损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 设置优化器（随机梯度下降）
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 开始训练
# 训练流程
def train_loop(dataloader, model, loss_fn, optimizer):
    # 检测数据集长度
    size = len(dataloader.dataset)
    # 将模型设置为训练模式（只对部分网络生效，如dropout）
    model.train()
    # 开始将数据加载进神经网络
    for batch, (X, y) in enumerate(dataloader):
        # 生成预测值与loss值
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 每计算完100个包print一个结果
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss值: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试流程
def test_loop(dataloader, model, loss_fn):
    # 将模型设置为测试模式
    model.eval()
    # 计算正确率相关参数
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 将梯度置零
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # 使用预测值与target对比，判断预测正确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"测试结果: \n 正确率: {(100*correct):>0.1f}%, 平均loss: {test_loss:>8f} \n")


# 正式开始训练
epochs = 20
for t in range(epochs):
    print(f"第 {t+1} 轮\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

torch.save(model, "model.pth")
print("模型已保存")
print("训练完成!")


