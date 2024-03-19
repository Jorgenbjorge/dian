import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义RNN网络
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载Fashion-MNIST数据集
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transformer,
    download=True,
)
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=transformer,
)

# 定义数据加载器
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
model = RNNModel(input_size, hidden_size, num_layers, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.view(-1, 28, 28)  # 将图像展平为序列
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.6f}')

# 计算指标
def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs.data, dim=1)

    tp = ((predicted == labels) & (labels == 1)).sum().item()
    tn = ((predicted != labels) & (labels == 0)).sum().item()
    fp = ((predicted != labels) & (labels == 1)).sum().item()
    fn = ((predicted == labels) & (labels == 0)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score


# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.view(-1, 28, 28)
        outputs = model(images)
        accuracy, precision, recall, f1_score = calculate_metrics(outputs, labels)
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

print(f'Accuracy: {100 * accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1_score:.2f}')
print(f'Overall Accuracy: {100 * correct / total:.2f}%')

# 可视化
sample_images = next(iter(test_loader))[0][:16]
sample_outputs = model(sample_images.view(-1, 28, 28))
_, predicted = torch.max(sample_outputs, 1)

fig = plt.figure(figsize=(10, 4))
for idx in range(16):
    ax = fig.add_subplot(2, 8, idx + 1, xticks=[], yticks=[])
    ax.imshow(sample_images[idx].squeeze(), cmap='gray')
    ax.set_title(f'Predicted: {predicted[idx].item()}')
plt.show()