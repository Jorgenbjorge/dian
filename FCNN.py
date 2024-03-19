import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 构建全连接神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 将训练集中的数据转换为计算机可读取的样式，并令其分布符合正态
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='../datasets/mnist',
    train=True,
    transform=transformer,
    download=True,
)
test_dataset = datasets.MNIST(
    root='../datasets/mnist',
    train=False,
    transform=transformer,
)

# 定义数据加载器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=64,
                         shuffle=False)

# 初始化模型
model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 训练
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.6f}')


# 计算指标
def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)

    tp = ((predicted == labels) & (labels == 1)).sum().item()
    tn = ((predicted != labels) & (labels == 0)).sum().item()
    fp = ((predicted != labels) & (labels == 1)).sum().item()
    fn = ((predicted == labels) & (labels == 0)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score


# 测试
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        accuracy, precision, recall, f1_score = calculate_metrics(outputs, labels)
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

print(f'Accuracy: {100 * accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1_score:.2f}')
print(f'Overall Accuracy: {100 * correct / total:.2f}%')